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
        if self.combination_method == "neural":
            if model_path is None:
                logger.warning("No model_path provided for ltr")
                self.combination_method = "harmonic"
            else:
                self._load_neural_model(model_path)
                logger.info(f"Loaded neural ltr combiner from {model_path} on {self.device}")
    
    def _load_neural_model(self, model_path: str):
        n_features = len(self.weight_functions) + 1 
        self.neural_model = NeuralCombiner(n_features).to(self.device)
        self.neural_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.neural_model.eval()
    
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
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            combined = self.neural_model(features_tensor).cpu().numpy().flatten()
        return combined
    
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
    def __init__(self, n_features: int, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralCombiner(n_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(
        self,
        train_data: List[Tuple[np.ndarray, int]],
        val_data: List[Tuple[np.ndarray, int]],
        epochs: int = 50,
        batch_size: int = 32
    ):
        print(f"Training on {len(train_data)} samples, validating on {len(val_data)}")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                features = np.array([x for x, _ in batch])
                labels = np.array([y for _, y in batch])
                features = torch.FloatTensor(features).to(self.device)
                labels = torch.FloatTensor(labels).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                val_loss = self._validate(val_data)
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss = {train_loss/len(train_data):.4f}, "
                      f"Val Loss = {val_loss:.4f}")
    
    def _validate(self, val_data):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            features = np.array([x for x, _ in val_data])
            labels = np.array([y for _, y in val_data])
            features = torch.FloatTensor(features).to(self.device)
            labels = torch.FloatTensor(labels).to(self.device)
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            val_loss = loss.item()
        
        return val_loss
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")