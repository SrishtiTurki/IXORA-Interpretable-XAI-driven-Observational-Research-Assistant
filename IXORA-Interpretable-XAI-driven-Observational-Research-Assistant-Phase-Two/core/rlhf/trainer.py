# core/rlhf/trainer.py - DOMAIN-AWARE REWARD MODEL TRAINER

import json
import os
import time
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

logger = logging.getLogger("core.rlhf.trainer")

@dataclass
class TrainingConfig:
    """Configuration for reward model training"""
    batch_size: int = 16
    learning_rate: float = 1e-5
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    
    @classmethod
    def for_domain(cls, domain: str) -> 'TrainingConfig':
        """Get domain-specific training config"""
        domain = domain.lower()
        if domain == "biomed":
            return cls(
                batch_size=8,
                learning_rate=5e-6,
                num_epochs=15,
                warmup_steps=50
            )
        elif domain == "computerscience":
            return cls(
                batch_size=16,
                learning_rate=2e-5,
                num_epochs=10,
                warmup_steps=100
            )
        return cls()

class DomainAwarePreferenceDataset(Dataset):
    """Dataset for domain-aware preference learning"""
    
    def __init__(self, filepath: str = "logs/rlhf_feedback.jsonl", 
                 domain: Optional[str] = None,
                 min_samples: int = 10):
        """
        Args:
            filepath: Path to feedback JSONL file
            domain: If specified, only use samples from this domain
            min_samples: Minimum samples required per domain
        """
        self.pairs: List[Tuple[str, str, str]] = []  # (good, bad, domain)
        self.domain_stats: Dict[str, int] = {}
        self.domain = domain.lower() if domain else None
        
        try:
            self._load_and_process_data(filepath, min_samples)
            logger.info(f"✅ Loaded dataset with {len(self.pairs)} preference pairs")
            logger.info(f"🌐 Domain distribution: {self.domain_stats}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing dataset: {e}", exc_info=True)
            self.pairs = []
    
    def _load_and_process_data(self, filepath: str, min_samples: int):
        """Load and process feedback data"""
        feedback_by_query: Dict[str, List[dict]] = {}
        
        # Load and group feedback
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    query_hash = item.get("query_hash", "unknown")
                    domain = item.get("domain", "unknown").lower()
                    
                    # Filter by domain if specified
                    if self.domain and domain != self.domain:
                        continue
                        
                    # Track domain statistics
                    self.domain_stats[domain] = self.domain_stats.get(domain, 0) + 1
                    
                    if query_hash not in feedback_by_query:
                        feedback_by_query[query_hash] = []
                    feedback_by_query[query_hash].append(item)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"⚠️ Error parsing feedback: {e}")
                    continue
        
        # Create preference pairs
        self._create_preference_pairs(feedback_by_query, min_samples)
    
    def _create_preference_pairs(self, feedback_by_query: Dict[str, List[dict]], 
                               min_samples: int):
        """Create preference pairs from feedback data"""
        for query_hash, feedbacks in feedback_by_query.items():
            if len(feedbacks) < 2:
                continue  # Need at least 2 responses for comparison
            
            # Group by preference
            goods = [f for f in feedbacks if f.get("preference") == 1]
            bads = [f for f in feedbacks if f.get("preference") == 0]
            
            # Create balanced pairs
            if goods and bads:
                domain = feedbacks[0].get("domain", "unknown").lower()
                # Limit pairs per query to avoid imbalance
                max_pairs = min(len(goods), len(bads), 3)  # Max 3 pairs per query
                
                for i in range(max_pairs):
                    self.pairs.append((
                        goods[i % len(goods)]["response_text"],
                        bads[i % len(bads)]["response_text"],
                        domain
                    ))
        
        # Filter domains with insufficient samples
        self._filter_low_frequency_domains(min_samples)
    
    def _filter_low_frequency_domains(self, min_samples: int):
        """Remove domains with too few samples"""
        domain_counts = {}
        filtered_pairs = []
        
        # Count samples per domain
        for good, bad, domain in self.pairs:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Filter pairs
        for good, bad, domain in self.pairs:
            if domain_counts[domain] >= min_samples:
                filtered_pairs.append((good, bad, domain))
        
        self.pairs = filtered_pairs
        self.domain_stats = {}
        for _, _, domain in self.pairs:
            self.domain_stats[domain] = self.domain_stats.get(domain, 0) + 1
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self.pairs[idx]
    
    def split_train_test(self, test_size: float = 0.2):
        """Split into train and test sets"""
        if not self.pairs:
            return DomainAwarePreferenceDataset(), DomainAwarePreferenceDataset()
        
        # Group by domain for stratified split
        domains = [domain for _, _, domain in self.pairs]
        train_idx, test_idx = train_test_split(
            range(len(self.pairs)),
            test_size=test_size,
            random_state=42,
            stratify=domains
        )
        
        train_data = DomainAwarePreferenceDataset()
        test_data = DomainAwarePreferenceDataset()
        
        train_data.pairs = [self.pairs[i] for i in train_idx]
        test_data.pairs = [self.pairs[i] for i in test_idx]
        
        # Update domain stats
        for good, bad, domain in train_data.pairs:
            train_data.domain_stats[domain] = train_data.domain_stats.get(domain, 0) + 1
        for good, bad, domain in test_data.pairs:
            test_data.domain_stats[domain] = test_data.domain_stats.get(domain, 0) + 1
        
        return train_data, test_data

class RewardModelTrainer:
    """Domain-aware reward model trainer"""
    
    def __init__(self, domain: Optional[str] = None):
        self.domain = domain.lower() if domain else None
        self.config = TrainingConfig.for_domain(domain) if domain else TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = f"logs/training/{domain or 'all'}_{int(time.time())}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        logger.info(f"Initialized trainer for domain: {domain or 'all'}")
        logger.info(f"Using device: {self.device}")
    
    async def train(self, train_data: DomainAwarePreferenceDataset,
                   test_data: Optional[DomainAwarePreferenceDataset] = None):
        """Train the reward model"""
        from .reward_model import get_reward_model, save_reward_model
        
        # Initialize model
        model = get_reward_model(self.domain or "default")
        model.to(self.device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create data loaders
        train_loader = self._create_data_loader(train_data, shuffle=True)
        test_loader = self._create_data_loader(test_data) if test_data else None
        
        # Training loop
        global_step = 0
        best_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Forward pass
                good_logits = model(batch["good"])
                bad_logits = model(batch["bad"])
                
                # Compute loss (higher score for preferred responses)
                loss = -torch.nn.functional.logsigmoid(good_logits - bad_logits).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                
                # Log metrics
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % self.config.log_interval == 0:
                    self._log_metrics(
                        epoch=epoch,
                        step=global_step,
                        loss=loss.item(),
                        phase="train"
                    )
                
                # Evaluate on test set
                if test_loader and global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(model, test_loader)
                    self._log_metrics(
                        epoch=epoch,
                        step=global_step,
                        phase="eval",
                        **eval_metrics
                    )
                    
                    # Save best model
                    if eval_metrics["accuracy"] > best_accuracy:
                        best_accuracy = eval_metrics["accuracy"]
                        save_reward_model(self.domain or "default")
                        logger.info(f"💾 Saved best model with accuracy: {best_accuracy:.4f}")
                
                # Save checkpoint
                if global_step % self.config.save_interval == 0:
                    save_reward_model(self.domain or "default")
                    logger.info(f"💾 Saved checkpoint at step {global_step}")
            
            # Log epoch metrics
            avg_loss = epoch_loss / len(train_loader)
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {avg_loss:.4f}")
        
        # Save final model
        save_reward_model(self.domain or "default")
        logger.info("✅ Training completed")
        
        return model
    
    def evaluate(self, model, data_loader):
        """Evaluate the model on the given data loader"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                good_logits = model(batch["good"])
                bad_logits = model(batch["bad"])
                
                # Compute loss
                loss = -torch.nn.functional.logsigmoid(good_logits - bad_logits).mean()
                total_loss += loss.item()
                
                # Compute accuracy
                correct += (good_logits > bad_logits).sum().item()
                total += len(good_logits)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(data_loader) if data_loader else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _create_data_loader(self, dataset: DomainAwarePreferenceDataset, 
                           shuffle: bool = False) -> DataLoader:
        """Create a data loader from a dataset"""
        if not dataset or len(dataset) == 0:
            return None
            
        def collate_fn(batch):
            good_texts = [item[0] for item in batch]
            bad_texts = [item[1] for item in batch]
            domains = [item[2] for item in batch]
            
            return {
                "good": good_texts,
                "bad": bad_texts,
                "domain": domains
            }
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
    
    def _log_metrics(self, epoch: int, step: int, phase: str, **metrics):
        """Log metrics to console and TensorBoard"""
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{name}", value, step)
        
        # Log to console
        log_msg = f"[{phase.upper()}] Epoch {epoch+1} | Step {step}"
        for name, value in metrics.items():
            log_msg += f" | {name}: {value:.4f}"
        
        logger.info(log_msg)

async def train_reward_model(domain: Optional[str] = None):
    """Train a reward model (optionally for a specific domain)"""
    try:
        logger.info(f"� Starting reward model training for domain: {domain or 'all'}")
        
        # Load dataset
        dataset = DomainAwarePreferenceDataset(domain=domain)
        
        if len(dataset) < 10:
            logger.warning(f"⚠️ Not enough training data. Need at least 10 samples, got {len(dataset)}")
            return False
        
        # Split into train/test
        train_data, test_data = dataset.split_train_test(test_size=0.2)
        
        logger.info(f"📊 Training samples: {len(train_data)}")
        logger.info(f"📊 Test samples: {len(test_data)}")
        
        # Initialize and train
        trainer = RewardModelTrainer(domain)
        await trainer.train(train_data, test_data)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return False
        
        # Initialize model
        from core.rlhf.reward_model import get_reward_model, save_reward_model
        model = get_reward_model()
        model.train()
        
        # Training setup
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=2, factor=0.5, verbose=True
        )
        
        # Prepare data loaders
        train_loader = DataLoader(
            dataset.train_pairs,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Training loop
        best_loss = float('inf')
        early_stop_patience = 3
        patience_counter = 0
        
        for epoch in range(10):
            epoch_start = time.time()
            total_loss = 0
            model.train()
            
            for batch in train_loader:
                # Unpack batch (text1, text2, domain)
                chosen_texts = batch[0]
                rejected_texts = batch[1]
                
                # Skip empty batches
                if not chosen_texts or not rejected_texts:
                    continue
                    
                optimizer.zero_grad()
                
                # Forward pass
                chosen_rewards = model(chosen_texts)
                rejected_rewards = model(rejected_texts)
                
                # Bradley-Terry loss
                loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Calculate metrics
            avg_loss = total_loss / len(train_loader)
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Update learning rate
            scheduler.step(avg_loss)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/10 - "
                  f"Loss: {avg_loss:.4f} - "
                  f"Time: {epoch_time:.1f}s - "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                save_reward_model()
                print(f"💾 New best model saved with loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Final model save
        save_reward_model()
        print("✅ Training completed!")
        
        # Test if we have test data
        if dataset.test_pairs:
            print("\n🧪 Running evaluation...")
            model.eval()
            all_correct = 0
            all_total = 0
            domain_metrics = {}
            
            with torch.no_grad():
                test_loader = DataLoader(
                    dataset.test_pairs,
                    batch_size=8,
                    shuffle=False
                )
                
                for batch in test_loader:
                    chosen_texts = batch[0]
                    rejected_texts = batch[1]
                    domains = batch[2]
                    
                    if not chosen_texts or not rejected_texts:
                        continue
                    
                    # Get model predictions
                    chosen_scores = model(chosen_texts)
                    rejected_scores = model(rejected_texts)
                    
                    # Calculate accuracy
                    batch_correct = (chosen_scores > rejected_scores).sum().item()
                    batch_total = len(chosen_texts)
                    
                    all_correct += batch_correct
                    all_total += batch_total
                    
                    # Track metrics by domain
                    for i, domain in enumerate(domains):
                        if i >= len(chosen_scores):
                            continue
                            
                        if domain not in domain_metrics:
                            domain_metrics[domain] = {'correct': 0, 'total': 0}
                            
                        domain_metrics[domain]['total'] += 1
                        if chosen_scores[i] > rejected_scores[i]:
                            domain_metrics[domain]['correct'] += 1
            
            # Log overall accuracy
            if all_total > 0:
                accuracy = all_correct / all_total
                print(f"\n📊 Test Results:")
                print(f"- Overall Accuracy: {accuracy:.2%} ({all_correct}/{all_total})")
                
                # Log domain-specific metrics
                print("\n🌐 Domain-wise Accuracy:")
                for domain, metrics in domain_metrics.items():
                    if metrics['total'] > 0:
                        domain_acc = metrics['correct'] / metrics['total']
                        print(f"- {domain.capitalize()}: {domain_acc:.2%} "
                              f"({metrics['correct']}/{metrics['total']})")
                        writer.add_scalar(f'Accuracy/{domain}', domain_acc, 0)
            
            writer.add_scalar('Accuracy/overall', accuracy, 0)
            writer.close()
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Training failed: {e}")
        print("\nStack trace:")
        traceback.print_exc()
        return False