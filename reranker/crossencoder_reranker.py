"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Simple Cross-Encoder reranker implementation for FEVER fact verification.
    Uses a pre-trained sequence classification model to score, referenced from A2 implementation done in class
    
Code:
"""

import torch
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .reranker import BaseReranker
from src.config import LOGGER as logger


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_path: str = "reranker/models/naver/trecdl22-crossencoder-debertav3",
        device: str = None,
        max_length: int = 512,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        
        self.max_length = max_length
        self.batch_size = batch_size
        
        logger.info(f"Initializing CrossEncoder Reranker on {self.device}")
        logger.info(f"Model: {self.model_path}")
        
        self._load_model()
    
    def _load_model(self):
        try:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
            except FileNotFoundError as e:
                logger.indo(f"Error finding local model, downloading from HuggingFace hub")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path
                )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Cross encoder model loaded successfully")
        except Exception as e:
            print(f"Error loading Cross Encoder model: {str(e)}")
            raise
        
            
    
    def compute_score(self, query: str, document: str) -> float:
        pair = [[f"Query: {query}\n", f"Document: {document}\n"]]
        
        inputs = self.tokenizer(
            pair,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits.squeeze(-1).item()
        return score
    
    def batch_compute_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        if not documents:
            return []
        
        all_scores = []
        
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            
            pairs = [[f"Query: {query}\n", f"Document: {doc}\n"] 
                     for doc in batch_docs]
            
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
            
            if not isinstance(batch_scores, list):
                batch_scores = [batch_scores]
            
            all_scores.extend(batch_scores)
        
        return all_scores
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str]],
        top_k: int = None
    ) -> List[Tuple[str, str, float]]:
        if not documents:
            return []
        
        if top_k is None:
            top_k = len(documents)
        top_k = min(top_k, len(documents))
        
        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")
        
        doc_ids, doc_texts = zip(*documents)
        
        scores = self.batch_compute_scores(query, list(doc_texts))
        
        scored_docs = list(zip(doc_ids, doc_texts, scores))
        
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        results = scored_docs[:top_k]
        
        logger.debug(f"Reranking complete. Returning top {len(results)} documents")
        
        return results
