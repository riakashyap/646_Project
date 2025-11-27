"""
Copyright:

  Copyright © 2025 Ananya-Jha-code
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
trainer.py — Finetunes naver on FEVER

Loss used: PairwiseRankingLoss from lossfunc.py
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW
)
from tqdm import tqdm
from pathlib import Path

from lossfunc import PairwiseRankingLoss
from e2rank_reranker import E2RankReranker  
from src.config import (
    CLAIMS_PATH, QRELS_PATH, RANKLISTS_PATH,
    PAGES_DIR
)


class FeverRerankDataset(Dataset):

    def __init__(
        self,
        claims_path: Path,
        qrels_path: Path,
        ranklist_path: Path,
        pages_dir: Path,
        max_negatives: int = 1
    ):
        self.claims_path = claims_path
        self.qrels_path = qrels_path
        self.ranklist_path = ranklist_path
        self.pages_dir = pages_dir
        self.max_negatives = max_negatives

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
        file_path = self.pages_dir / f"{pid}.txt"
        if not file_path.exists():
            return ""
        return file_path.read_text(errors="ignore")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class RerankerTrainer:

    def __init__(
        self,
        model_type: str = "pairwise",    
        model_name: str = "naver/trecdl22-crossencoder-debertav3",
        device: str = None,
        lr: float = 1e-5,
        use_layerwise: bool = True
    ):
     
        self.model_type = model_type.strip().lower()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


        if self.model_type == "pairwise":
            print("[Trainer] Using PAIRWISE")

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1
            ).to(self.device)

        elif self.model_type == "e2rank":
            print("[Trainer] Using E2RANK")
            self.reranker = E2RankReranker(
                model_path=model_name,
                device=self.device,
                use_layerwise=use_layerwise
            )
            self.model = self.reranker.model

        else:
            raise ValueError("model_type must be 'pairwise' or 'e2rank'.")

 
        self.loss_fn = PairwiseRankingLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        print(f"Loaded model: {self.model_type} @ {model_name}")

    def _score_ce(self, queries, docs):
        inp = self.tokenizer(
            queries, docs,
            padding=True, truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        out = self.model(**inp)
        return out.logits.squeeze(-1)

    def _score_e2rank(self, queries, docs):
        scores = []
        for q, d in zip(queries, docs):
            score = self.reranker.compute_score(q, d)
            scores.append(score)
        return torch.tensor(scores, device=self.device)

    def _score(self, queries, docs):
        if self.model_type == "pairwise":
            return self._score_ce(queries, docs)
        else:
            return self._score_e2rank(queries, docs)

    
    def train(self, dataset: Dataset, batch_size=4, num_epochs=2):

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            total_loss = 0

            for batch in tqdm(loader):
                queries, pos_docs, neg_docs = zip(*batch)

                pos_scores = self._score(queries, pos_docs)
                neg_scores = self._score(queries, neg_docs)

                loss = self.loss_fn(pos_scores, neg_scores)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch Loss: {total_loss / len(loader):.4f}")

        print("\nTraining complete.")

    def save(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model_type == "pairwise":
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), output_dir / "pytorch_model.bin")
            self.tokenizer.save_pretrained(output_dir)

        print(f"Saved model to {output_dir}")