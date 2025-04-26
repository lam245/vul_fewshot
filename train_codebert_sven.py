import json
import torch
import numpy as np
import logging
import os
import random
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sven_binary_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Set random seed for reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

class SVENBinaryDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512, model_type="codebert"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
        # Preprocess data to create binary classification samples
        self.processed_examples = []
        for example in examples:
            # Add vulnerable sample (before) with label 0
            self.processed_examples.append({
                "code": example["func_src_before"],
                "label": 1  # vulnerable code
            })
            
            # Add fixed sample (after) with label 1
            self.processed_examples.append({
                "code": example["func_src_after"],
                "label": 0  # fixed code
            })
        
        logger.info(f"Created {len(self.processed_examples)} samples for binary classification")
        
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx):
        example = self.processed_examples[idx]
        
        # Tokenize single code snippet
        inputs = self.tokenizer(
            example["code"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Extract and flatten tensors
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(example["label"], dtype=torch.long)
        }
        
        # For CodeT5, add decoder_input_ids
        if self.model_type == "codet5":
            # For classification, we don't need meaningful decoder inputs
            # Just use a simple start token
            decoder_input_ids = torch.zeros((1), dtype=torch.long)
            result["decoder_input_ids"] = decoder_input_ids
        
        return result

def load_sven_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logger.warning(f"Could not parse line: {line}")
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Custom wrapper for CodeT5 classification
class CodeT5ForSequenceClassification(torch.nn.Module):
    def __init__(self, checkpoint, num_labels=2, device="cuda"):
        super().__init__()
        # Load T5 model directly
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            checkpoint, 
            trust_remote_code=True
        ).to(device)
        
        # Add classification head
        self.classifier = torch.nn.Linear(
            self.t5.config.d_model, 
            num_labels
        ).to(device)
        
        self.device = device
        self.num_labels = num_labels
    
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        # T5 requires decoder_input_ids
        if decoder_input_ids is None:
            # Use a default decoder input if none provided
            batch_size = input_ids.size(0)
            decoder_input_ids = torch.zeros(
                (batch_size, 1), 
                dtype=torch.long, 
                device=self.device
            )
        
        # Get encoder outputs from T5
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use mean pooling of encoder hidden states for classification
        # This is one approach; another would be to use the first token
        hidden_states = encoder_outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # Apply the classifier
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def load_model_and_tokenizer(model_type, device="cuda"):
    """
    Load the specified model type and its tokenizer
    """
    if model_type == "codebert":
        checkpoint = "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = RobertaForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=2  # Binary classification
        ).to(device)
    elif model_type == "codet5":
        checkpoint = "Salesforce/codet5p-220m-bimodal"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        
        # Create our custom CodeT5 classification model
        model = CodeT5ForSequenceClassification(checkpoint, num_labels=2, device=device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose either 'codebert' or 'codet5'")
    
    logger.info(f"Loaded {model_type} model from {checkpoint}")
    return model, tokenizer

def main():
    # Set up configuration
    config = {
        "model_type": "codet5",  # Options: "codebert" or "codet5"
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_path": "/home/itvkist/code/old_code/vul_check/vul_fewshot/sven_train.jsonl",
        "output_dir": "./sven_model_output",
        "max_length": 512,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "epochs": 20,
        "warmup_steps": 500,
        "save_steps": 1000,
        "eval_steps": 1000,
        "seed": 42,
        "test_size": 0.2,
        "val_size": 0.1
    }
    
    # Adjust output directory based on model type
    config["output_dir"] = f"./sven_{config['model_type']}_binary_model"
    
    # Set random seed
    set_random_seed(config["seed"])
    
    # Load data
    logger.info(f"Loading data from {config['data_path']}")
    data = load_sven_data(config["data_path"])
    
    # Load model and tokenizer based on the selected type
    model, tokenizer = load_model_and_tokenizer(config["model_type"], config["device"])
    
    # Split data into train, validation, and test sets
    train_val_data, test_data = train_test_split(
        data, test_size=config["test_size"], random_state=config["seed"]
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=config["val_size"]/(1-config["test_size"]), 
        random_state=config["seed"]
    )
    
    logger.info(f"Train set: {len(train_data)} examples (will be doubled for binary classification)")
    logger.info(f"Validation set: {len(val_data)} examples (will be doubled for binary classification)")
    logger.info(f"Test set: {len(test_data)} examples (will be doubled for binary classification)")
    
    # Create datasets
    train_dataset = SVENBinaryDataset(train_data, tokenizer, config["max_length"], config["model_type"])
    val_dataset = SVENBinaryDataset(val_data, tokenizer, config["max_length"], config["model_type"])
    test_dataset = SVENBinaryDataset(test_data, tokenizer, config["max_length"], config["model_type"])
    
    logger.info(f"Final train set: {len(train_dataset)} samples")
    logger.info(f"Final validation set: {len(val_dataset)} samples")
    logger.info(f"Final test set: {len(test_dataset)} samples")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        learning_rate=config["learning_rate"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model
    logger.info(f"Starting training with {config['model_type']} model")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")
    
    # Save model, tokenizer, and label mapping
    logger.info(f"Saving model to {config['output_dir']}")
    if config["model_type"] == "codebert":
        # Standard save for HuggingFace models
        trainer.save_model()
    else:
        # For CodeT5 with custom wrapper, save component parts
        os.makedirs(config["output_dir"], exist_ok=True)
        # Save T5 model
        model.t5.save_pretrained(os.path.join(config["output_dir"], "t5_model"))
        # Save classifier
        torch.save(model.classifier.state_dict(), os.path.join(config["output_dir"], "classifier.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(config["output_dir"])
    
    # Save model configuration and label information for future inference
    model_config = {
        "model_type": config["model_type"],
        "label_info": {
            "0": "fixed_code",
            "1": "vulnerable_code"
        }
    }
    
    with open(os.path.join(config["output_dir"], "model_config.json"), "w") as f:
        json.dump(model_config, f)
    
    # Return test results
    return test_results

if __name__ == "__main__":
    main()