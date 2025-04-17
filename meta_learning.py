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
            AdamW,
            get_linear_schedule_with_warmup
        )
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from sklearn.model_selection import train_test_split
        import copy
        import pandas as pd
        from tqdm import tqdm
        from config import *
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("meta_learning_vul_detection.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Set random seed for reproducibility
        def set_random_seed(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logger.info(f"Random seed set to {seed}")

        def filter_datasets_by_matching_vulnerability_types(sven_data, big_vul_df):
            """
            Filter both datasets to include only samples with matching vulnerability types.
            This version directly compares CWE IDs without using a mapping, using case-insensitive matching.
            
            Args:
                sven_data: List of dictionaries containing SVEN data
                big_vul_df: DataFrame containing BigVul data
            
            Returns:
                filtered_sven_data: Filtered SVEN data
                filtered_big_vul_df: Filtered BigVul data
                common_vuln_types: List of common vulnerability types
            """
            # Extract unique vulnerability types from SVEN (converted to lowercase)
            sven_vuln_types = set()
            sven_vuln_type_map = {}  # Map to preserve original case
            for example in sven_data:
                if "vul_type" in example and example["vul_type"]:
                    lowercase_type = example["vul_type"].lower()
                    sven_vuln_types.add(lowercase_type)
                    sven_vuln_type_map[lowercase_type] = example["vul_type"]  # Remember original case
            
            # Extract unique CWE IDs from BigVul (converted to lowercase)
            big_vul_cwe_ids = set()
            big_vul_cwe_map = {}  # Map to preserve original case
            if "CWE ID" in big_vul_df.columns:
                for cwe_id in big_vul_df["CWE ID"].dropna().unique():
                    lowercase_cwe = str(cwe_id).lower()
                    big_vul_cwe_ids.add(lowercase_cwe)
                    big_vul_cwe_map[lowercase_cwe] = cwe_id  # Remember original case
            else:
                logger.warning("CWE ID column not found in BigVul dataset")
                return [], big_vul_df.iloc[0:0], []  # Return empty datasets
            
            # Match using lowercase versions
            common_vuln_types_lower = sven_vuln_types.intersection(big_vul_cwe_ids)
            
            # Convert back to original case (using BigVul's case as the standard)
            common_vuln_types_original = [big_vul_cwe_map.get(vt, vt) for vt in common_vuln_types_lower]
            
            logger.info(f"Found {len(common_vuln_types_original)} common vulnerability types: {common_vuln_types_original}")
            
            if not common_vuln_types_original:
                logger.warning("No common vulnerability types found between datasets")
                return [], big_vul_df.iloc[0:0], []
            
            # Create a set of lowercase common types for efficient lookup
            common_vuln_types_lower_set = set(common_vuln_types_lower)
            
            # Filter SVEN data (case-insensitive matching)
            filtered_sven_data = [
                example for example in sven_data 
                if "vul_type" in example and example["vul_type"] and example["vul_type"].lower() in common_vuln_types_lower_set
            ]
            
            # Filter BigVul data (case-insensitive matching)
            filtered_big_vul_df = big_vul_df[big_vul_df["CWE ID"].astype(str).str.lower().isin(common_vuln_types_lower_set)]
            
            logger.info(f"Filtered SVEN data: {len(filtered_sven_data)} samples")
            logger.info(f"Filtered BigVul data: {len(filtered_big_vul_df)} samples")
            
            return filtered_sven_data, filtered_big_vul_df, common_vuln_types_original

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
            
            # Check if "CWE ID" column exists
            if "CWE ID" not in df.columns:
                logger.warning("'CWE ID' column not found in BigVul dataset. Vulnerability type matching will not be possible.")
            
            # Ensure label is binary (0 for non-vulnerable, 1 for vulnerable)
            df['vul'] = df['vul'].astype(int)
            
            # Sample the dataset if requested
            if sample_size and sample_size < len(df):
                df = df.sample(sample_size, random_state=42)
            
            logger.info(f"Big-Vul dataset loaded with {len(df)} samples")
            logger.info(f"Vulnerable samples: {df['vul'].sum()} ({df['vul'].mean()*100:.2f}%)")
            
            return df

        def compute_metrics(labels, preds):
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        class MAMLForVulnerabilityDetection:
            def __init__(
                self, 
                model_name, 
                inner_lr=0.01, 
                outer_lr=0.001, 
                num_inner_steps=5,
                num_outer_steps=1000,
                meta_batch_size=4,
                device=None
            ):
                self.model_name = model_name
                self.inner_lr = inner_lr
                self.outer_lr = outer_lr
                self.num_inner_steps = num_inner_steps
                self.num_outer_steps = num_outer_steps
                self.meta_batch_size = meta_batch_size
                self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Initialize model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = RobertaForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2  # Binary classification
                ).to(self.device)
                
                # Initialize optimizer
                self.meta_optimizer = AdamW(self.model.parameters(), lr=self.outer_lr)
                
            def inner_loop_update(self, support_dataloader):
                """Perform inner loop updates on a task (support set)"""
                # Clone the model for this task
                task_model = copy.deepcopy(self.model)
                task_optimizer = AdamW(task_model.parameters(), lr=self.inner_lr)
                
                # Perform gradient steps on support set
                task_model.train()
                for _ in range(self.num_inner_steps):
                    for batch in support_dataloader:
                        # Move tensors to device
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        
                        # Forward pass
                        outputs = task_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        
                        # Backward pass and update
                        task_optimizer.zero_grad()
                        loss.backward()
                        task_optimizer.step()
                
                return task_model
            
            def evaluate_model(self, model, dataloader):
                """Evaluate model on given dataloader"""
                model.eval()
                all_preds = []
                all_labels = []
                total_loss = 0
                
                with torch.no_grad():
                    for batch in dataloader:
                        # Move tensors to device
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        
                        # Forward pass
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        total_loss += loss.item()
                        
                        # Get predictions
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        labels = labels.cpu().numpy()
                        
                        all_preds.extend(preds)
                        all_labels.extend(labels)
                
                # Compute metrics
                metrics = compute_metrics(all_labels, all_preds)
                metrics['loss'] = total_loss / len(dataloader)
                
                return metrics
            
            def meta_train(self, big_vul_tasks, sven_task, val_dataloader, num_epochs=1):
                """Meta-training using Big-Vul tasks"""
                logger.info("Starting meta-training phase")
                best_val_f1 = 0
                best_model_state = None
                
                for epoch in range(num_epochs):
                    # Shuffle tasks for this epoch
                    random.shuffle(big_vul_tasks)
                    
                    # Process each meta-batch
                    for i in range(0, len(big_vul_tasks), self.meta_batch_size):
                        meta_batch = big_vul_tasks[i:i + self.meta_batch_size]
                        
                        # Initialize meta-batch gradient
                        self.meta_optimizer.zero_grad()
                        
                        # Accumulate gradients over tasks in meta-batch
                        meta_batch_loss = 0
                        
                        for task_support_dl, task_query_dl in meta_batch:
                            # Perform inner loop update
                            updated_task_model = self.inner_loop_update(task_support_dl)
                            
                            # Evaluate on query set to get meta-gradient
                            updated_task_model.train()
                            query_loss = 0
                            
                            for batch in task_query_dl:
                                # Move tensors to device
                                input_ids = batch["input_ids"].to(self.device)
                                attention_mask = batch["attention_mask"].to(self.device)
                                labels = batch["labels"].to(self.device)
                                
                                # Forward pass
                                outputs = updated_task_model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels
                                )
                                
                                # Accumulate loss
                                task_loss = outputs.loss
                                query_loss += task_loss
                            
                            # Get average loss for this task
                            task_query_loss = query_loss / len(task_query_dl)
                            meta_batch_loss += task_query_loss
                        
                        # Get average loss for meta-batch
                        meta_loss = meta_batch_loss / len(meta_batch)
                        
                        # Compute meta-gradient and update meta-parameters
                        meta_loss.backward()
                        self.meta_optimizer.step()
                        
                        # Evaluate on validation set every few steps
                        if (i // self.meta_batch_size) % 10 == 0:
                            # Adapt to SVEN task with current meta-parameters
                            support_dl, _ = sven_task
                            adapted_model = self.inner_loop_update(support_dl)
                            
                            # Evaluate on SVEN validation set
                            val_metrics = self.evaluate_model(adapted_model, val_dataloader)
                            logger.info(f"Epoch {epoch+1}, Step {i//self.meta_batch_size}: Val F1: {val_metrics['f1']:.4f}, Loss: {val_metrics['loss']:.4f}")
                            
                            # Save best model
                            if val_metrics['f1'] > best_val_f1:
                                best_val_f1 = val_metrics['f1']
                                best_model_state = copy.deepcopy(self.model.state_dict())
                                logger.info(f"New best validation F1: {best_val_f1:.4f}")
                
                # Load best model
                if best_model_state:
                    self.model.load_state_dict(best_model_state)
                    logger.info(f"Loaded best model with validation F1: {best_val_f1:.4f}")
                
                return best_val_f1
            
            def fine_tune(self, train_dataloader, val_dataloader, num_epochs=3, lr=5e-5, warmup_steps=100):
                """Fine-tune on target dataset after meta-training"""
                logger.info("Starting fine-tuning phase")
                
                optimizer = AdamW(self.model.parameters(), lr=lr)
                total_steps = len(train_dataloader) * num_epochs
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=500, 
                    num_training_steps=total_steps
                )
                
                best_val_f1 = 0
                best_model_state = None
                
                for epoch in range(num_epochs):
                    # Training
                    self.model.train()
                    train_loss = 0
                    
                    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
                        # Move tensors to device
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        
                        # Forward pass
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        train_loss += loss.item()
                        
                        # Backward pass and update
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                    
                    avg_train_loss = train_loss / len(train_dataloader)
                    
                    # Validation
                    val_metrics = self.evaluate_model(self.model, val_dataloader)
                    
                    logger.info(f"Epoch {epoch+1}/{num_epochs}")
                    logger.info(f"Train Loss: {avg_train_loss:.4f}")
                    logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
                    logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
                    logger.info(f"Val F1: {val_metrics['f1']:.4f}")
                    logger.info(f"Val Precision: {val_metrics['precision']:.4f}")
                    logger.info(f"Val Recall: {val_metrics['recall']:.4f}")
                    
                    # Save best model
                    if val_metrics['f1'] > best_val_f1:
                        best_val_f1 = val_metrics['f1']
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        logger.info(f"New best validation F1: {best_val_f1:.4f}")
                
                # Load best model
                if best_model_state:
                    self.model.load_state_dict(best_model_state)
                    logger.info(f"Loaded best model with validation F1: {best_val_f1:.4f}")
                
                return best_val_f1
            
            def save_model(self, output_dir):
                """Save model and tokenizer"""
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                
                label_info = {
                    "0": "vulnerable_code",
                    "1": "fixed_code"
                }
                
                with open(os.path.join(output_dir, "label_info.json"), "w") as f:
                    json.dump(label_info, f)
                
                logger.info(f"Model saved to {output_dir}")

        def create_meta_tasks(df, tokenizer, n_tasks=10, k_shot=8, max_length=512):
            """Create meta-learning tasks from Big-Vul dataset"""
            tasks = []
            
            for _ in range(n_tasks):
                # Ensure we have enough samples of each class
                pos_samples = df[df['vul'] == 1]
                neg_samples = df[df['vul'] == 0]
                
                # If we don't have enough samples, use sampling with replacement
                pos_replace = len(pos_samples) < k_shot * 2
                neg_replace = len(neg_samples) < k_shot * 2
                
                pos_samples = pos_samples.sample(k_shot * 2, replace=pos_replace)
                neg_samples = neg_samples.sample(k_shot * 2, replace=neg_replace)
                
                # Create support and query sets
                support_df = pd.concat([
                    pos_samples.iloc[:k_shot],
                    neg_samples.iloc[:k_shot]
                ])
                
                query_df = pd.concat([
                    pos_samples.iloc[k_shot:],
                    neg_samples.iloc[k_shot:]
                ])
                
                # Create datasets
                support_dataset = VulnerabilityDataset(support_df, tokenizer, max_length=max_length, is_bigvul=True)
                query_dataset = VulnerabilityDataset(query_df, tokenizer, max_length=max_length, is_bigvul=True)
                
                # Create dataloaders
                support_dataloader = DataLoader(support_dataset, batch_size=4, shuffle=True)
                query_dataloader = DataLoader(query_dataset, batch_size=4)
                
                tasks.append((support_dataloader, query_dataloader))
            
            logger.info(f"Created {len(tasks)} meta-learning tasks")
            return tasks

        def create_sven_task(sven_data, tokenizer, max_length=512):
            """Create a task from SVEN dataset for adaptation"""
            # Split data for support set (for adaptation) and test
            train_data, test_data = train_test_split(
                sven_data, test_size=0.2, random_state=42
            )
            
            # Create datasets
            support_dataset = VulnerabilityDataset(train_data, tokenizer, max_length=max_length)
            test_dataset = VulnerabilityDataset(test_data, tokenizer, max_length=max_length)
            
            # Create dataloaders
            support_dataloader = DataLoader(support_dataset, batch_size=4, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=4)
            
            return (support_dataloader, test_dataloader)

        def main():
            # Set configuration
            config = config
            
            # Set random seed
            set_random_seed(config["seed"])
            
            # Load datasets
            logger.info(f"Loading SVEN data from {config['sven_data_path']}")
            sven_data = load_sven_data(config["sven_data_path"])
            
            logger.info(f"Loading Big-Vul data from {config['big_vul_data_path']}")
            big_vul_df = load_big_vul_dataset(config["big_vul_data_path"], sample_size=config["big_vul_sample_size"])
            
            # Filter datasets by matching vulnerability types if enabled
            if config["use_vulnerability_type_matching"]:
                logger.info("Filtering datasets by matching vulnerability types")
                sven_data, big_vul_df, common_vuln_types = filter_datasets_by_matching_vulnerability_types(
                    sven_data, big_vul_df
                )
                
                # Log common vulnerability types
                logger.info(f"Common vulnerability types: {common_vuln_types}")
                logger.info(f"Filtered SVEN data size: {len(sven_data)}")
                logger.info(f"Filtered BigVul data size: {len(big_vul_df)}")
                
                # Check if we have enough data to proceed
                if len(sven_data) == 0 or len(big_vul_df) == 0:
                    logger.error("Insufficient data after filtering. Exiting.")
                    return
                
                # Update output directory to reflect filtered dataset
                config["output_dir"] = f"{config['output_dir']}_filtered_by_vul_type"
            
            # Initialize MAML
            maml = MAMLForVulnerabilityDetection(
                model_name=config["model_name"],
                inner_lr=config["inner_lr"],
                outer_lr=config["outer_lr"],
                num_inner_steps=config["num_inner_steps"],
                meta_batch_size=config["meta_batch_size"],
                device=device
            )
            
            # Create meta-tasks from Big-Vul
            big_vul_tasks = create_meta_tasks(
                big_vul_df, 
                maml.tokenizer, 
                n_tasks=config["n_meta_tasks"], 
                k_shot=config["k_shot"], 
                max_length=config["max_length"]
            )
            
            # Create target task from SVEN
            sven_task = create_sven_task(sven_data, maml.tokenizer, max_length=config["max_length"])
            
            # Split SVEN data for fine-tuning and evaluation
            train_val_data, test_data = train_test_split(sven_data, test_size=0.2, random_state=config["seed"])
            train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=config["seed"])
            
            # Create datasets for fine-tuning and evaluation
            train_dataset = VulnerabilityDataset(train_data, maml.tokenizer, max_length=config["max_length"])
            val_dataset = VulnerabilityDataset(val_data, maml.tokenizer, max_length=config["max_length"])
            test_dataset = VulnerabilityDataset(test_data, maml.tokenizer, max_length=config["max_length"])
            
            # Create dataloaders
            train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"])
            test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])
            
            # Meta-training phase
            logger.info("Starting meta-training phase")
            maml.meta_train(
                big_vul_tasks, 
                sven_task, 
                val_dataloader, 
                num_epochs=config["meta_epochs"]
            )
            
            # Fine-tuning phase
            logger.info("Starting fine-tuning phase")
            maml.fine_tune(
                train_dataloader, 
                val_dataloader, 
                num_epochs=config["ft_epochs"], 
                lr=config["ft_lr"]
            )
            
            # Evaluate on test set
            logger.info("Evaluating on test set")
            test_metrics = maml.evaluate_model(maml.model, test_dataloader)
            logger.info(f"Test metrics: {test_metrics}")
            
            # Save the model
            maml.save_model(config["output_dir"])
            
            # Baseline comparison: evaluate a model without meta-learning
            logger.info("Training baseline model (without meta-learning)")
            baseline_model = RobertaForSequenceClassification.from_pretrained(
                config["model_name"],
                num_labels=2
            ).to(device)
            
            baseline_optimizer = AdamW(baseline_model.parameters(), lr=config["ft_lr"])
            baseline_total_steps = len(train_dataloader) * config["ft_epochs"]
            baseline_scheduler = get_linear_schedule_with_warmup(
                baseline_optimizer, 
                num_warmup_steps=100, 
                num_training_steps=baseline_total_steps
            )
            
            # Train baseline model
            baseline_model.train()
            for epoch in range(config["ft_epochs"]):
                epoch_loss = 0
                
                for batch in tqdm(train_dataloader, desc=f"Baseline Epoch {epoch+1}/{config['ft_epochs']}"):
                    # Move tensors to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Forward pass
                    outputs = baseline_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    epoch_loss += loss.item()
                    
                    # Backward pass and update
                    baseline_optimizer.zero_grad()
                    loss.backward()
                    baseline_optimizer.step()
                    baseline_scheduler.step()
                
                # Log progress
                logger.info(f"Baseline Epoch {epoch+1}: Avg loss = {epoch_loss/len(train_dataloader):.4f}")
            
            # Evaluate baseline
            baseline_model.eval()
            baseline_preds = []
            baseline_labels = []
            
            with torch.no_grad():
                for batch in test_dataloader:
                    # Move tensors to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Forward pass
                    outputs = baseline_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Get predictions
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = labels.cpu().numpy()
                    
                    baseline_preds.extend(preds)
                    baseline_labels.extend(labels)
            
            baseline_metrics = compute_metrics(baseline_labels, baseline_preds)
            
            # Compare results
            logger.info("Results comparison:")
            logger.info(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
            logger.info(f"Baseline F1: {baseline_metrics['f1']:.4f}")
            logger.info(f"Meta-Learning Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Meta-Learning F1: {test_metrics['f1']:.4f}")
            
            # Save comparison results
            result_summary = {
                "baseline": baseline_metrics,
                "meta_learning": test_metrics
            }
            
            with open(os.path.join(config["output_dir"], "results_comparison.json"), "w") as f:
                json.dump(result_summary, f, indent=4)
            
            return test_metrics

        if __name__ == "__main__":
            main()