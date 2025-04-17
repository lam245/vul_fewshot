import json
import logging
import os
import random
import argparse
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import torch
import numpy as np
from transformers import (RobertaConfig, RobertaForSequenceClassification, 
                          RobertaTokenizer, AdamW, get_linear_schedule_with_warmup)
import higher
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class VulnerabilityDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512, is_bigvul=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_bigvul = is_bigvul
        
        # Store processed examples
        self.processed_examples = []
        
        if is_bigvul:
            # Process Big-Vul format
            for _, row in examples.iterrows():
                self.processed_examples.append({
                    "code": str(row['func_before']),
                    "label": int(row['vul'])
                })
        else:
            # Process SVEN format - create binary classification samples
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
        
        logger.info(f"Created {len(self.processed_examples)} samples for {'Big-Vul' if is_bigvul else 'SVEN'} dataset")
    
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx):
        example = self.processed_examples[idx]
        
        # Tokenize code snippet
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

def load_big_vul_dataset(dataset_path, sample_size=None):
    """Load the Big-Vul dataset with its specific format"""
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Ensure important columns exist
    required_columns = ['func_before', 'vul']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")
    
    # Ensure label is binary (0 for non-vulnerable, 1 for vulnerable)
    df['vul'] = df['vul'].astype(int)
    
    # Sample the dataset if requested
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    logger.info(f"Big-Vul dataset loaded with {len(df)} samples")
    logger.info(f"Vulnerable samples: {df['vul'].sum()} ({df['vul'].mean()*100:.2f}%)")
    
    return df

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

def prepare_dataloaders(datasets, batch_size):
    """
    Prepare dataloaders for each dataset with random sampling
    
    Args:
        datasets: Dictionary mapping dataset names to Dataset objects
        batch_size: Batch size for each dataloader
        
    Returns:
        Dictionary mapping dataset names to DataLoader objects
    """
    dataloaders = {}
    for name, dataset in datasets.items():
        sampler = RandomSampler(dataset)
        dataloaders[name] = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size
        )
    return dataloaders

def evaluate_model(model, dataloader, device):
    """Evaluate model on the given dataloader"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss, logits = outputs[:2]
            
            # Collect predictions and labels
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            total_loss += loss.item()
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Calculate precision, recall, and F1 for the positive class (vulnerable code)
    true_positives = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
    false_positives = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))
    false_negatives = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def maml_train_vulnerability(args, train_datasets, eval_datasets, model, device):
    """
    Implement MAML training for vulnerability detection
    
    Args:
        args: Arguments containing training parameters
        train_datasets: Dictionary mapping dataset names to training datasets
        eval_datasets: Dictionary mapping dataset names to evaluation datasets
        model: The model to train
        device: Device to train on
    """
    # Create optimizer for the meta model
    optimizer = AdamW(model.parameters(), lr=args.meta_lr)
    
    # Prepare dataloaders
    train_dataloaders = prepare_dataloaders(train_datasets, args.batch_size)
    
    # Initialize best metrics
    best_dev_f1 = 0.0
    best_model_state = None
    
    # Training loop
    logger.info("***** Running MAML training for vulnerability detection *****")
    logger.info(f"  Num training epochs = {args.num_epochs}")
    logger.info(f"  Meta learning rate = {args.meta_lr}")
    logger.info(f"  Task learning rate = {args.task_lr}")
    logger.info(f"  Batch size = {args.batch_size}")
    
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        # Create an epoch iterator for progress tracking
        epoch_iterator = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for _ in epoch_iterator:
            meta_loss = 0
            
            # Outer loop - iterate over each dataset/task
            for dataset_name, dataloader in train_dataloaders.items():
                # Get a batch from the current dataset
                try:
                    batch = next(iter(dataloader))
                except StopIteration:
                    # Reinitialize dataloader if we've gone through all batches
                    train_dataloaders[dataset_name] = DataLoader(
                        train_datasets[dataset_name],
                        sampler=RandomSampler(train_datasets[dataset_name]),
                        batch_size=args.batch_size
                    )
                    batch = next(iter(train_dataloaders[dataset_name]))
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Split batch for support (training) and query (testing)
                batch_size = batch["input_ids"].size(0)
                split_idx = batch_size // 2
                
                support_batch = {
                    "input_ids": batch["input_ids"][:split_idx],
                    "attention_mask": batch["attention_mask"][:split_idx],
                    "labels": batch["labels"][:split_idx]
                }
                
                query_batch = {
                    "input_ids": batch["input_ids"][split_idx:],
                    "attention_mask": batch["attention_mask"][split_idx:],
                    "labels": batch["labels"][split_idx:]
                }
                
                # Create inner optimizer for task-specific adaptation
                inner_optimizer = torch.optim.SGD(model.parameters(), lr=args.task_lr)
                
                # Inner loop adaptation using higher library
                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fast_model, diffopt):
                    # Adapt the model on the support set (inner loop)
                    fast_model.train()
                    support_outputs = fast_model(**support_batch)
                    support_loss = support_outputs[0]
                    diffopt.step(support_loss)
                    
                    # Evaluate the adapted model on the query set
                    query_outputs = fast_model(**query_batch)
                    query_loss = query_outputs[0]
                    
                    # Accumulate meta loss
                    meta_loss += query_loss
            
            # Backward pass and optimization step for meta parameters
            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_iterator.set_postfix({"meta_loss": meta_loss.item()})
            epoch_loss += meta_loss.item()
            global_step += 1
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1} - Average Meta Loss: {epoch_loss/args.steps_per_epoch:.4f}")
        
        # Evaluate on dev sets
        logger.info("Evaluating on development sets...")
        eval_results = {}
        for name, dataset in eval_datasets.items():
            eval_dataloader = DataLoader(dataset, batch_size=args.batch_size)
            results = evaluate_model(model, eval_dataloader, device)
            eval_results[name] = results
            
            logger.info(f"  {name} - F1: {results['f1']:.4f}, Accuracy: {results['accuracy']:.4f}")
        
        # Save best model based on SVEN dev F1 score
        if "sven_dev" in eval_results and eval_results["sven_dev"]["f1"] > best_dev_f1:
            best_dev_f1 = eval_results["sven_dev"]["f1"]
            best_model_state = model.state_dict().copy()
            
            # Save the best model
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_dev_f1,
                }, output_path)
                logger.info(f"Best model saved to {output_path} with F1: {best_dev_f1:.4f}")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_dev_f1

def fine_tune_on_target(args, model, train_dataset, dev_dataset, device):
    """Fine-tune the meta-trained model on the target dataset (SVEN)"""
    # Create optimizer and dataloader
    optimizer = AdamW(model.parameters(), lr=args.fine_tune_lr)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Training loop
    logger.info("***** Fine-tuning on target dataset *****")
    logger.info(f"  Num fine-tuning epochs = {args.fine_tune_epochs}")
    
    best_dev_f1 = 0.0
    best_model_state = None
    
    for epoch in range(args.fine_tune_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs[0]
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1} - Average Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        # Evaluate on dev set
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
        results = evaluate_model(model, dev_dataloader, device)
        
        logger.info(f"  Dev - F1: {results['f1']:.4f}, Accuracy: {results['accuracy']:.4f}")
        
        # Save best model based on F1 score
        if results["f1"] > best_dev_f1:
            best_dev_f1 = results["f1"]
            best_model_state = model.state_dict().copy()
            
            # Save the best fine-tuned model
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, "best_fine_tuned_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_dev_f1,
                }, output_path)
                logger.info(f"Best fine-tuned model saved to {output_path} with F1: {best_dev_f1:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_dev_f1

def main():
    parser = argparse.ArgumentParser()
    # Dataset paths
    parser.add_argument("--big_vul_path", type=str, required=True,
                        help="Path to Big-Vul dataset CSV file")
    parser.add_argument("--sven_train_path", type=str, required=True,
                        help="Path to SVEN training data JSON file")
    
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base",
                        help="Pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./vulnerability_models_maml",
                        help="Directory to save models")
    
    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of meta-learning epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--meta_lr", type=float, default=5e-5, help="Meta learning rate")
    parser.add_argument("--task_lr", type=float, default=1e-4, help="Task adaptation learning rate")
    
    # Fine-tuning hyperparameters
    parser.add_argument("--fine_tune", action="store_true", 
                        help="Whether to fine-tune on target dataset after meta-training")
    parser.add_argument("--fine_tune_epochs", type=int, default=3, 
                        help="Number of fine-tuning epochs")
    parser.add_argument("--fine_tune_lr", type=float, default=2e-5,
                        help="Fine-tuning learning rate")
    
    # BigVul sampling
    parser.add_argument("--big_vul_samples", type=int, default=10000,
                        help="Number of samples to use from Big-Vul dataset")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load and prepare datasets
    logger.info("Loading datasets...")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    # Load Big-Vul dataset
    big_vul_df = load_big_vul_dataset(args.big_vul_path, sample_size=args.big_vul_samples)
    
    # Split Big-Vul dataset into train/dev
    big_vul_train = big_vul_df.sample(frac=0.9, random_state=args.seed)
    big_vul_dev = big_vul_df.drop(big_vul_train.index)
    
    # Load SVEN datasets
    sven_train_data = load_sven_data(args.sven_train_path)
    sven_train_data, sven_test_data = train_test_split(sven_train_data, test_size=0.2, random_state=42)

# Then split the remaining data to create a dev set (25% of the remaining data, which is 20% of the original)
    sven_train_data, sven_dev_data = train_test_split(sven_train_data, test_size=0.25, random_state=42)
    
    # Create PyTorch datasets
    train_datasets = {
        "big_vul": VulnerabilityDataset(big_vul_train, tokenizer, args.max_seq_length, is_bigvul=True),
        "sven": VulnerabilityDataset(sven_train_data, tokenizer, args.max_seq_length, is_bigvul=False)
    }
    
    eval_datasets = {
        "big_vul_dev": VulnerabilityDataset(sven_dev_data, tokenizer, args.max_seq_length, is_bigvul=True),
        "sven_dev": VulnerabilityDataset(sven_test_data, tokenizer, args.max_seq_length, is_bigvul=False),
    }
    
    test_dataset = VulnerabilityDataset(sven_test_data, tokenizer, args.max_seq_length, is_bigvul=False)
    
    # Initialize model
    logger.info(f"Initializing model from {args.model_name}...")
    config = RobertaConfig.from_pretrained(args.model_name, num_labels=2)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.to(device)
    
    # Meta-training with MAML
    logger.info("Starting meta-training...")
    model, best_meta_f1 = maml_train_vulnerability(args, train_datasets, eval_datasets, model, device)
    
    # Fine-tuning on the target dataset (SVEN) if specified
    if args.fine_tune:
        logger.info("Fine-tuning on SVEN dataset...")
        model, best_fine_tune_f1 = fine_tune_on_target(
            args, model, train_datasets["sven"], eval_datasets["sven_dev"], device
        )
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_results = evaluate_model(model, test_dataloader, device)
    
    logger.info("Test Results:")
    logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"  Precision: {test_results['precision']:.4f}")
    logger.info(f"  Recall: {test_results['recall']:.4f}")
    logger.info(f"  F1: {test_results['f1']:.4f}")
    
    # Save the final model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_results': test_results,
        }, output_path)
        logger.info(f"Final model saved to {output_path}")

if __name__ == "__main__":
    main()