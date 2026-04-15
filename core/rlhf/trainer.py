# core/rlhf/trainer.py - IMPROVED VERSION

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger("core.rlhf.trainer")

class PreferenceDataset(Dataset):
    def __init__(self, filepath="logs/rlhf_feedback.jsonl", min_samples=5):
        self.pairs = []
        
        try:
            if not Path(filepath).exists():
                logger.warning(f"No feedback file found at {filepath}")
                return
            
            # Load all feedback
            feedback_items = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        feedback_items.append(json.loads(line))
            
            if len(feedback_items) < min_samples:
                logger.info(f"Not enough feedback samples: {len(feedback_items)} < {min_samples}")
                return
            
            # Group by query_hash to find comparable responses
            feedback_by_query = {}
            for item in feedback_items:
                query_hash = item.get("query_hash", "unknown")
                if query_hash not in feedback_by_query:
                    feedback_by_query[query_hash] = []
                feedback_by_query[query_hash].append(item)
            
            # Create preference pairs (chosen vs rejected)
            for query_hash, items in feedback_by_query.items():
                if len(items) < 2:
                    continue
                
                # Separate good and bad responses
                good_items = [i for i in items if i.get("preference", 0) >= 0.7]
                bad_items = [i for i in items if i.get("preference", 0) < 0.3]
                
                if good_items and bad_items:
                    # Create pairs (good vs bad)
                    for good in good_items[:2]:  # Limit to 2 good per query
                        for bad in bad_items[:2]:  # Limit to 2 bad per query
                            self.pairs.append((
                                good["response_text"],
                                bad["response_text"]
                            ))
            
            logger.info(f"Created {len(self.pairs)} preference pairs from {len(feedback_items)} feedback items")
            
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            self.pairs = []
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

async def train_reward_model():
    """Train the reward model on collected feedback"""
    try:
        # Load dataset
        dataset = PreferenceDataset(min_samples=3)
        
        if len(dataset) < 3:
            logger.warning("⚠️ Not enough training data (need at least 5 pairs)")
            return False
        
        # Initialize model
        from core.rlhf.reward_model import get_reward_model
        model = get_reward_model()
        model.train()
        
        # Split dataset
        train_pairs, val_pairs = train_test_split(
            dataset.pairs, test_size=0.2, random_state=42
        )
        
        if len(train_pairs) < 2:
            logger.warning("Not enough training pairs after split")
            return False
        
        # Training setup
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Convert to DataLoader format
        train_data = []
        for chosen, rejected in train_pairs:
            train_data.append((chosen, 1.0))  # Preferred response
            train_data.append((rejected, 0.0))  # Rejected response
        
        # Simple training loop
        epochs = 5
        batch_size = 4
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batches = 0
            
            # Shuffle data
            np.random.shuffle(train_data)
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                if len(batch) < 2:
                    continue
                
                texts = [item[0] for item in batch]
                labels = torch.tensor([item[1] for item in batch], dtype=torch.float32).unsqueeze(1)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(texts)
                
                # Compute loss
                loss = loss_fn(predictions, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            if batches > 0:
                avg_loss = total_loss / batches
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save model
        from core.rlhf.reward_model import save_reward_model
        save_reward_model()
        
        # Quick validation
        if val_pairs:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                
                for chosen, rejected in val_pairs[:10]:  # Test on first 10
                    chosen_score = model([chosen]).item()
                    rejected_score = model([rejected]).item()
                    
                    if chosen_score > rejected_score:
                        correct += 1
                    total += 1
                
                if total > 0:
                    accuracy = correct / total
                    logger.info(f"Validation Accuracy: {accuracy:.2%}")
        
        logger.info("✅ Reward model trained successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False