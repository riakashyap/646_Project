"""
Copyright:

  Copyright © 2025 uchuuronin
  Copyright © 2025 Ananya-Jha-code

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    E2Rank reranker implementation using layer-wise progressive reranking.
    Based on the paper "E2Rank: Efficient and Effective Layer-wise Reranking"
    Called model found @ https://github.com/caesar-one/e2rank
    HuggingFace Model for cross-encoder: https://huggingface.co/naver/trecdl22-crossencoder-debertav3

Code:
"""

import torch
from typing import List, Tuple, Dict, Optional
from .reranker import BaseReranker
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from .grouped_debertav2 import GroupedDebertaV2ForSequenceClassification, remap_state_dict
from src.config import LOGGER as logger


class E2RankReranker(BaseReranker):
    def __init__(
        self,
        model_path: str = "reranker/models/naver/trecdl22-crossencoder-debertav3",
        device: str = None,
        max_length: int = 512,
        use_layerwise: bool = True,
        reranking_block_map: Dict[int, int] = None,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)

        self.max_length = max_length
        self.use_layerwise = use_layerwise

        # Progressively reduce candidates: Layers : Top-K Docs to keep
        if reranking_block_map is None:
            self.reranking_block_map = {
                8: 50,
                16: 28,
                24: 10
            }
        else:
            self.reranking_block_map = reranking_block_map

        logger.info(f"Device: {self.device} (is CUDA available: {torch.cuda.is_available()})")
        logger.info(f"Initializing E2RankReranker on {self.device}")
        logger.info(f"Layerwise reranking: {self.use_layerwise}")
        if self.use_layerwise:
            logger.info(f"Reranking strategy: {self.reranking_block_map}")

        self._load_model()

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            config = AutoConfig.from_pretrained(self.model_path)

            self.model = GroupedDebertaV2ForSequenceClassification(config)

            try:
                state_dict = torch.load(
                    f"{self.model_path}/pytorch_model.bin",
                    map_location=self.device
                )
                
                state_dict = remap_state_dict(state_dict)
                self.model.load_state_dict(state_dict, strict=False)
            except FileNotFoundError:
                print(
                    f"Could not find pytorch_model.bin at {self.model_path}. "
                    "Loading from HuggingFace Hub..."
                )
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path
                )
                state_dict = remap_state_dict(base_model.state_dict()) 
                self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()

            logger.info("E2Rank model loaded successfully")
        except Exception as e:
            print(f"Error loading E2Rank model: {str(e)}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str]],
        top_k: int = None
    ) -> List[Tuple[str, str, float]]:

        if not documents:
            return []

        if top_k is None and self.use_layerwise:
            top_k = min(self.reranking_block_map.values()) if self.reranking_block_map else len(documents)
        elif top_k is None:
            top_k = len(documents)

        top_k = min(top_k, len(documents))

        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")

        doc_ids, doc_texts = zip(*documents)

        pairs = [[f"Query: {query}\n", f"Document: {text}\n"] for text in doc_texts]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            if self.use_layerwise and len(documents) > 10:
                # Use progressive layerwise reranking for efficiency
                reranked_indices = self.model.layerwise_rerank(
                    **inputs,
                    reranking_block_map=self.reranking_block_map
                )
                reranked_indices = reranked_indices.cpu().tolist()[:top_k]

                final_pairs = [[f"Query: {query}\n", f"Document: {doc_texts[idx]}\n"] for idx in reranked_indices]
                final_inputs = self.tokenizer(
                    final_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                final_outputs = self.model(**final_inputs)
                final_logits = final_outputs.logits.squeeze(-1)
                final_scores = final_logits.cpu().tolist()

                # Handle single document case (k=1)
                if not isinstance(final_scores, list):
                    final_scores = [final_scores]

                results = [
                    (doc_ids[idx], doc_texts[idx], score)
                    for idx, score in zip(reranked_indices, final_scores)
                ]
            else:
                # Standard full-model reranking for small sets
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)
                scores = logits.cpu().tolist()

                # Sort by score descending
                scored_docs = list(zip(doc_ids, doc_texts, scores))
                scored_docs.sort(key=lambda x: x[2], reverse=True)
                results = scored_docs[:top_k]
        logger.debug(f"Reranking complete. Returning top {len(results)} results")
        return results

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

        pairs = [[f"Query: {query}\n", f"Document: {doc}\n"] for doc in documents]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()

        return scores if isinstance(scores, list) else [scores]
