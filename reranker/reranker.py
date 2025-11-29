"""
Copyright:

  Copyright © 2025 uchuuronin
  

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Base reranker class defining the interface for all reranker implementations.

Code:
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import torch

class BaseReranker(ABC):
    """
    Abstract base class for reranker implementations.
    Reranker model should inherit from this class and implement method.
    """
    
    def __init__(self, model_path: str = None, device: str = None, **kwargs):
        """
        Initialize the reranker.
        
        Args:
            model_path: Path or name of the model to load
            device: Device defauted to use whatever is available (cuda/cpu)
        """
        self.model_path = model_path
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: List[Tuple[str, str]], 
        top_k: int = None
    ) -> List[Tuple[str, str, float]]:
        """
        Rerank a list of documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of (doc_id, doc_text) tuples from initial retrieval
            top_k: Number of top documents to return. If None, returns all reranked.
            
        Returns:
            List of (doc_id, doc_text, relevance_score) tuples, sorted by relevance (highest first)
        """
        pass
    
    @abstractmethod
    def compute_score(self, query: str, document: str) -> float:
        """
        Compute relevance score for a single query-document pair.
        
        Args:
            query: The search query
            document: The document text
            
        Returns:
            Relevance score (higher = more relevant)
        """
        pass
    
    def batch_compute_scores(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[float]:
        """
        Compute relevance scores for multiple query-document pairs efficiently.
        
        Args:
            query: The search query
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        return [self.compute_score(query, doc) for doc in documents]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_path}, device={self.device})"
