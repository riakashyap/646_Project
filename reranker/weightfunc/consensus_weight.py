"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Weight function that prioritises evidence supported by multiple independent sources.
    - Boosts documents that have text similarity to other top-scoring documents.
    - NEI if equal docs Support and Refute.
    
Code:
"""
from typing import List, Tuple
from collections import defaultdict
import re
from .weightfunc import BaseWeightFunction
from sentence_transformers import SentenceTransformer
from src.config import LOGGER as logger
import math
from collections import Counter
import numpy as np
import torch

class ConsensusWeightFunction(BaseWeightFunction):
  def __init__(self, min_similarity: float = 0.3, multiplier: float = 1.3, sim_method: str = "tfidf", device: str = None, embedding_model: SentenceTransformer = 'all-MiniLM-L6-v2'):
    super().__init__()
    self.min_similarity = min_similarity
    self.multiplier = multiplier
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device
      
    self.sim_method = sim_method # supported options are "tfidf" (better if no gpu) or "dense" (if gpu) 
    if self.sim_method == "dense":
      self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
      logger.info(f"Loaded embedding embedding_model: {embedding_model} on {device}")
      

  def _extract_keywords(self, text: str) -> list:  
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = [word for word in text.split() if len(word) > 2]  
    return words
  
  def _compute_tfidf(self, text1: str, text2: str, all_texts: List[str]) -> float:
    tokens1 = self._extract_keywords(text1)  # Now a list
    tokens2 = self._extract_keywords(text2)  # Now a list
    if not tokens1 or not tokens2:
      return 0.0
    
    all_tokens = [self._extract_keywords(t) for t in all_texts]
    doc_freq = Counter()
    for tokens in all_tokens:
      doc_freq.update(set(tokens))  
    num_docs = len(all_texts)
    
    def get_tfidf_vector(tokens):
      tf = Counter(tokens)
      tfidf = {}
      for term, count in tf.items():
        tf_weight = count / len(tokens) if len(tokens) > 0 else 0
        idf_weight = math.log(num_docs / (doc_freq[term] + 1))
        tfidf[term] = tf_weight * idf_weight
      return tfidf
    
    vec1 = get_tfidf_vector(tokens1)
    vec2 = get_tfidf_vector(tokens2)
    
    common_terms = set(vec1.keys()) & set(vec2.keys())
    if not common_terms:
      return 0.0
    
    dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
      return 0.0
    
    return dot_product / (norm1 * norm2)
  
  def _compute_dense(self, embeddings: np.ndarray, idx1: int, idx2: int) -> float:
    vec1 = embeddings[idx1]
    vec2 = embeddings[idx2]
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
  
  def compute_weights(
      self, 
      query: str,
      documents: List[Tuple[str, str, float]]
  ) -> List[float]:
    if len(documents) <= 1:
      return [1.0] * len(documents)
    
    doc_texts = [text for _, text, _ in documents]
    weights = []
    
    for doc_text in doc_texts:
      if self.sim_method == "tfidf":
        query_sim = self._compute_tfidf(query, doc_text, doc_texts + [query])
      elif self.sim_method == "dense":
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        doc_emb = self.embedding_model.encode([doc_text], convert_to_numpy=True, show_progress_bar=False)[0]
        
        dot = np.dot(query_emb, doc_emb)
        norm_q = np.linalg.norm(query_emb)
        norm_d = np.linalg.norm(doc_emb)
        query_sim = dot / (norm_q * norm_d) if norm_q > 0 and norm_d > 0 else 0.0
      
      if query_sim >= self.min_similarity:
        boost = 1.0 + (self.multiplier - 1.0) * query_sim
        weights.append(boost)
      else:
        weights.append(1.0)
    
    return weights
