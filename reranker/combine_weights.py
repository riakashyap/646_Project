"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Weight combiner that aggregates multiple weight functions.
    
    Combination strategies:
    - "harmonic": Default, F-measure style (penalizes low scores)
    - "additive": Simple average
    - "neural": LTR combination (requires training/gpu preferred)
    
Code:
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import numpy as np
from src.config import LOGGER as logger

class WeightCombiner:
    def __init__(
        self,
        weight_functions: List,
        combination_method: str = "harmonic",
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.weight_functions = weight_functions
        self.combination_method = combination_method
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.neural_model = None
        self.scaler = None
        if self.combination_method == "neural":
            if model_path is None:
                logger.warning("No model_path provided for ltr")
                self.combination_method = "harmonic"
            else:
                self._load_neural_model(model_path)
                logger.info(f"Loaded neural ltr combiner from {model_path} on {self.device}")
    
    def _load_neural_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        n_features = checkpoint.get('n_features', len(self.weight_functions) + 1)
        self.neural_model = NeuralCombiner(n_features).to(self.device)
        self.neural_model.load_state_dict(checkpoint['model_state_dict'])
        self.neural_model.eval()
        self.scaler = checkpoint.get('scaler', None)  # ← ADD THIS
        if self.scaler:
            logger.info("Loaded feature scaler from checkpoint")
    
    def _combine_harmonic(self, weights_matrix: np.ndarray) -> np.ndarray:
        n_funcs = weights_matrix.shape[1]
        n_docs = weights_matrix.shape[0]
        
        combined = np.zeros(n_docs)
        for i in range(n_docs):
            harmonic_sum = sum(1.0 / w if w > 0 else 1e9 for w in weights_matrix[i])
            combined[i] = n_funcs / harmonic_sum if harmonic_sum < 1e8 else 0.0
        
        return combined
    
    def _combine_additive(self, weights_matrix: np.ndarray) -> np.ndarray:
        return np.mean(weights_matrix, axis=1)
    
    def _combine_neural(self, weights_matrix: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        features = np.concatenate([weights_matrix, base_scores.reshape(-1, 1)], axis=1)
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            raw_logits = self.neural_model(features_tensor).cpu().numpy().flatten()
            # Map to [0.5, 1.5] range with 1.0 as neutral
            weights = 0.5 + torch.sigmoid(torch.FloatTensor(raw_logits)).numpy()
            
        return weights
    
    def compute_combined_weights(
        self,
        query: str,
        documents: List[Tuple[str, str, float]]
    ) -> List[float]:
        if not self.weight_functions:
            return [1.0] * len(documents)
        
        all_weights = []
        for weight_fn in self.weight_functions:
            weights = weight_fn.compute_weights(query, documents)
            all_weights.append(weights)
        
        weights_matrix = np.array(all_weights).T
        
        base_scores = np.array([score for _, _, score in documents])
        
        if self.combination_method == "harmonic":
            combined = self._combine_harmonic(weights_matrix)
        elif self.combination_method == "additive":
            combined = self._combine_additive(weights_matrix)
        elif self.combination_method == "neural":
            combined = self._combine_neural(weights_matrix, base_scores)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return combined.tolist()
    
    def apply(
        self,
        query: str,
        documents: List[Tuple[str, str, float]],
        alpha: float = 1.0
    ) -> List[Tuple[str, str, float]]:
        if not documents:
            return []
        
        combined_weights = self.compute_combined_weights(query, documents)
        
        weighted_docs = [
            (doc_id, doc_text, score * (1.0 + alpha * (weight - 1.0)))  # ← CHANGE THIS
            for (doc_id, doc_text, score), weight in zip(documents, combined_weights)
        ]
        
        weighted_docs.sort(key=lambda x: x[2], reverse=True)
        
        return weighted_docs

class NeuralCombiner(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 32):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    
class NeuralCombinerTrainer:    
    """Updated trainer with pairwise ranking loss"""
    def __init__(self, n_features: int, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralCombiner(n_features, hidden_dim=64).to(self.device)  # Bigger hidden
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MarginRankingLoss(margin=1.0)  # ← CHANGED FROM BCEWithLogitsLoss
        self.scaler = None
    
    def train(
        self,
        train_data: List[Tuple[np.ndarray, np.ndarray]],  # ← CHANGED: now pairs (pos, neg)
        val_data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 50,
        batch_size: int = 128  # ← Bigger batch
    ):
        print(f"Training on {len(train_data)} pairs, validating on {len(val_data)}")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                pos_features = np.array([pos for pos, _ in batch])
                neg_features = np.array([neg for _, neg in batch])
                
                pos_tensor = torch.FloatTensor(pos_features).to(self.device)
                neg_tensor = torch.FloatTensor(neg_features).to(self.device)
                
                self.optimizer.zero_grad()
                
                pos_scores = self.model(pos_tensor)
                neg_scores = self.model(neg_tensor)
                
                # We want pos_score > neg_score
                target = torch.ones(pos_scores.size(0)).to(self.device)
                loss = self.criterion(pos_scores, neg_scores, target)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_correct += (pos_scores > neg_scores).sum().item()
            
            avg_train_loss = train_loss / len(train_data)
            train_acc = train_correct / len(train_data)
            
            # Validation
            val_loss, val_acc = self._validate(val_data)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={avg_train_loss:.4f}, Acc={train_acc:.3f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.3f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def _validate(self, val_data):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            pos_features = np.array([pos for pos, _ in val_data])
            neg_features = np.array([neg for _, neg in val_data])
            
            pos_tensor = torch.FloatTensor(pos_features).to(self.device)
            neg_tensor = torch.FloatTensor(neg_features).to(self.device)
            
            pos_scores = self.model(pos_tensor)
            neg_scores = self.model(neg_tensor)
            
            target = torch.ones(pos_scores.size(0)).to(self.device)
            loss = self.criterion(pos_scores, neg_scores, target)
            
            val_loss = loss.item()
            val_correct = (pos_scores > neg_scores).sum().item()
        
        return val_loss, val_correct / len(val_data)
    
    def save_model(self, path: str):
        """Save model with metadata"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_features': self.model.net[0].in_features,
            'scaler': self.scaler
        }, path)
        print(f"Model saved to {path}")