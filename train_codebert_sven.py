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
    def __init__(self, examples, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(example["label"], dtype=torch.long)
        }

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

def main():
    # Set up configuration
    config = {
        "model_name": "microsoft/codebert-base",
        "data_path": "/home/coder/sven/sven.json",
        "output_dir": "./sven_codebert_binary_model",
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
    
    # Set random seed
    set_random_seed(config["seed"])
    
    # Load data
    logger.info(f"Loading data from {config['data_path']}")
    data = load_sven_data(config["data_path"])
    
    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer from {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = RobertaForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2  # Binary classification: 0 for vulnerable, 1 for fixed
    )
    
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
    train_dataset = SVENBinaryDataset(train_data, tokenizer, config["max_length"])
    val_dataset = SVENBinaryDataset(val_data, tokenizer, config["max_length"])
    test_dataset = SVENBinaryDataset(test_data, tokenizer, config["max_length"])
    
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
    logger.info("Starting training")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")
    
    # Save model, tokenizer, and label mapping
    logger.info(f"Saving model to {config['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    
    # Save label information for future inference
    label_info = {
        "0": "vulnerable_code",
        "1": "fixed_code"
    }
    with open(os.path.join(config["output_dir"], "label_info.json"), "w") as f:
        json.dump(label_info, f)
    
    # Return test results
    return test_results

if __name__ == "__main__":
    main()