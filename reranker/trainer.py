"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Training model to finetune naver/trecdl22-crossencoder-debertav3 on FEVER
    Base model was previously finetuned on MS MARCO 
    E2Rank Training Approach: Layerwise cross-entropy loss with KL divergence
    Alt approach: Pairwise ranking loss (From Assignment A2)
    
    Also look into: Adding weight functions here based on (additive? )
        1. temporal_data (recent>old facts)
        2. credibility of source (news>social media)
        3. consensus based on volume of docs (agreed vs disagreed evidence)
    
Code:
"""


# class RerankerTrainer:
#     def __init__(
#         self,
#         model_name: str = "naver/trecdl22-crossencoder-debertav3"
#     ):
#         # Initialize reranker trainer
        
# ## add train, validate, save model methods here
"""
trainer.py — Finetunes naver/trecdl22-crossencoder-debertav3 on FEVER
Uses PairwiseRankingLoss from lossfunc.py
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm

from lossfunc import PairwiseRankingLoss       
from rerank import BaseReranker                
from pathlib import Path

class FeverRerankDataset(Dataset):

    def __init__(
        self,
        claims_path: Path,
        qrels_path: Path,
        ranklist_path: Path,
        pages_dir: Path,
        max_negatives: int = 1
    ):
        self.pages_dir = pages_dir
        self.max_negatives = max_negatives

        self.claims = json.loads(Path(claims_path).read_text())
        self.qrels = json.loads(Path(qrels_path).read_text())
        self.ranklists = json.loads(Path(ranklist_path).read_text())

        self.samples = []

        print("Building FEVER triplets")

        for claim_id, claim_obj in self.claims.items():
            claim_text = claim_obj["claim"]

            pos_docs = self.qrels.get(claim_id, [])
            if not pos_docs:
                continue

            pos_doc_id = pos_docs[0]
            pos_doc_text = self._load_page(pos_doc_id)

            bm25_docs = self.ranklists.get(claim_id, [])
            neg_doc_ids = [d for d in bm25_docs if d != pos_doc_id]
            if not neg_doc_ids:
                continue

            neg_doc_ids = neg_doc_ids[:max_negatives]

            for neg_id in neg_doc_ids:
                neg_text = self._load_page(neg_id)
                if neg_text.strip():
                    self.samples.append((claim_text, pos_doc_text, neg_text))

        print(f"Built {len(self.samples)} FEVER triplets.")

    def _load_page(self, pid: str):
        page_path = self.pages_dir / f"{pid}.txt"
        if not page_path.exists():
            return ""
        return page_path.read_text(errors="ignore")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class RerankerTrainer:
    """
    Trains a CrossEncoder reranker using PairwiseRankingLoss.
    """

    def __init__(
        self,
        model_name: str = "naver/trecdl22-crossencoder-debertav3",
        device: str = None,
        lr: float = 1e-5
    ):
        # Device handling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer + cross-encoder model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1        # single relevance score
        ).to(self.device)

        # Pairwise ranking loss from lossfunc.py
        self.loss_fn = PairwiseRankingLoss()

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        print(f"Loaded cross-encoder: {model_name}")

    def _score_pair(self, queries, docs):
        encoded = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        outputs = self.model(**encoded)
        return outputs.logits.squeeze(-1)   # (batch,)

    def train(self, dataset: Dataset, batch_size=4, num_epochs=2):

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            total_loss = 0

            for batch in tqdm(loader):
                queries, pos_docs, neg_docs = zip(*batch)

                pos_scores = self._score_pair(queries, pos_docs)
                neg_scores = self._score_pair(queries, neg_docs)

                loss = self.loss_fn(pos_scores, neg_scores)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch Loss: {total_loss / len(loader):.4f}")

        print("Training complete.")


    def save(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Saved model to {output_dir}")