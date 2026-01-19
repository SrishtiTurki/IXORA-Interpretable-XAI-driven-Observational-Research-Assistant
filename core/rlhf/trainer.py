# core/rlhf/trainer.py - IMPROVED VERSION
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ImprovedPreferenceDataset(Dataset):
    def __init__(self, filepath="logs/rlhf_feedback.jsonl", min_samples=10):
        self.pairs = []
        
        # Load and group by query_hash
        feedback_by_query = {}
        
        with open(filepath, "r") as f:
            for line in f:
                item = json.loads(line)
                query_hash = item.get("query_hash", "unknown")
                if query_hash not in feedback_by_query:
                    feedback_by_query[query_hash] = []
                feedback_by_query[query_hash].append(item)
        
        # Create balanced pairs
        for query_hash, feedbacks in feedback_by_query.items():
            if len(feedbacks) < 2:
                continue  # Need at least 2 responses for comparison
            
            goods = [f for f in feedbacks if f.get("preference") == 1]
            bads = [f for f in feedbacks if f.get("preference") == 0]
            
            if goods and bads:
                # Create limited pairs (avoid combinatorial explosion)
                for good in goods[:2]:  # Max 2 good per query
                    for bad in bads[:2]:  # Max 2 bad per query
                        self.pairs.append((
                            good["response_text"],
                            bad["response_text"]
                        ))
        
        print(f"📊 Created {len(self.pairs)} balanced preference pairs")
        
        # Split train/test
        if len(self.pairs) > min_samples:
            train_pairs, test_pairs = train_test_split(
                self.pairs, test_size=0.2, random_state=42
            )
            self.train_pairs = train_pairs
            self.test_pairs = test_pairs
        else:
            self.train_pairs = self.pairs
            self.test_pairs = []

async def train_reward_model():
    """Improved training function"""
    try:
        # Load dataset
        dataset = ImprovedPreferenceDataset()
        
        if len(dataset.train_pairs) < 5:
            print("⚠️ Not enough training data")
            return False
        
        # Initialize model
        from core.rlhf.reward_model import get_reward_model
        model = get_reward_model()
        model.train()
        
        # Training setup
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
        train_loader = DataLoader(
            list(dataset.train_pairs),  # Convert to list for DataLoader
            batch_size=8,
            shuffle=True
        )
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(10):
            total_loss = 0
            model.train()
            
            for chosen_texts, rejected_texts in train_loader:
                optimizer.zero_grad()
                
                chosen_rewards = model(chosen_texts)
                rejected_rewards = model(rejected_texts)
                
                # Bradley-Terry loss
                loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "models/reward_model_best.pth")
        
        # Final save
        torch.save(model.state_dict(), "models/reward_model.pth")
        print("✅ Reward model trained and saved!")
        
        # Test if we have test data
        if dataset.test_pairs:
            model.eval()
            with torch.no_grad():
                test_loader = DataLoader(list(dataset.test_pairs), batch_size=8)
                correct = 0
                total = 0
                
                for chosen_texts, rejected_texts in test_loader:
                    chosen_scores = model(chosen_texts)
                    rejected_scores = model(rejected_texts)
                    correct += (chosen_scores > rejected_scores).sum().item()
                    total += len(chosen_texts)
                
                accuracy = correct / total if total > 0 else 0
                print(f"📊 Test Accuracy: {accuracy:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False