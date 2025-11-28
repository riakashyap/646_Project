"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Abstract base class for defining weighting functions.
    
Code:
"""
from typing import List, Tuple
from abc import ABC, abstractmethod
import torch

class BaseWeightFunction(ABC):
  """
  Abstract base class for weights for reranker implementations.
  Weight functions should inherit from this class and implement method.
  """
  
  def __init__(self, **kwargs):
    """
    Initialize the weight function.
    """
    pass
  
  @abstractmethod
  def compute_weights(
    self, 
    query: str, 
    documents: List[Tuple[str, str, float]]
  ) -> List[float]:
    """
    Compute weight multipliers.
    
    Args:
        query: The search query/claim
        documents: List of (doc_id, doc_text, base_score) tuples
        
    Returns:
        List of weight multipliers 
    """
    pass
  
  def apply(
    self,
    query: str,
    documents: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
      """
      Apply weighting to documents and return reweighted results.
      
      Args:
          query: The search query/claim
          documents: List of (doc_id, doc_text, base_score) tuples
          
      Returns:
          List of (doc_id, doc_text, weighted_score) tuples
      """
      if not documents:
          return []
      
      weights = self.compute_weights(query, documents)
      
      return [
          (doc_id, doc_text, score * weight)
          for (doc_id, doc_text, score), weight in zip(documents, weights)
      ]
