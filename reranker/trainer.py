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
"""
trainer.py — Finetunes naver on FEVER

Loss used: PairwiseRankingLoss or LayerwiseCEKLLoss
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW
)
from tqdm import tqdm
from pathlib import Path

from reranker.lossfunc import PairwiseRankingLoss, LayerwiseCEKLLoss
from reranker.e2rank_reranker import E2RankReranker
from src.config import (
    CLAIMS_PATH, QRELS_PATH, RANKLISTS_PATH,
    PAGES_DIR
)
from pyserini.search.lucene import LuceneSearcher
from src.config import INDEX_DIR


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

        # Initialize searcher for fast doc lookup
        try:
            self.searcher = LuceneSearcher(str(INDEX_DIR))
        except Exception as e:
            print(f"Warning: Could not initialize LuceneSearcher: {e}")
            self.searcher = None

        # Load Data
        with open(self.claims_path, 'r', encoding='utf-8') as f:
            self.claims = json.load(f)
        with open(self.qrels_path, 'r', encoding='utf-8') as f:
            self.qrels = json.load(f)
        with open(self.ranklist_path, 'r', encoding='utf-8') as f:
            self.ranklists = json.load(f)

        self.samples = []

        print("Building FEVER triplets")

        for claim_id, claim_obj in self.claims.items():
            claim_text = claim_obj["claim"]

            pos_docs = self.qrels.get(claim_id, [])
            if not pos_docs:
                continue

            pos_doc_id = pos_docs[0]  # Take first relevant doc
            pos_doc_text = self._load_page(pos_doc_id)
            if not pos_doc_text:
                continue

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
        """Load page content from Index using Pyserini"""
        if not self.searcher:
            return ""
        try:
            doc = self.searcher.doc(pid)
            if doc:
                try:
                    # Try parsing as JSON if stored that way
                    json_content = json.loads(doc.raw())
                    return json_content.get('contents', doc.contents())
                except:
                    return doc.contents()
            return ""
        except Exception:
            return ""

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
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.model_type == "pairwise":
            print("[Trainer] Using PAIRWISE")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1
            ).to(self.device)
            self.loss_fn = PairwiseRankingLoss()

        elif self.model_type == "e2rank":
            print("[Trainer] Using E2RANK")
            self.reranker = E2RankReranker(
                model_path=model_name,
                device=self.device,
                use_layerwise=use_layerwise
            )
            self.model = self.reranker.model
            self.loss_fn = LayerwiseCEKLLoss()

        else:
            raise ValueError("model_type must be 'pairwise' or 'e2rank'.")

        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        print(f"Loaded model: {self.model_type} @ {model_name}")

    def train(self, dataset: Dataset, batch_size=4, num_epochs=2):

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            total_loss = 0

            for batch in tqdm(loader):
                self.optimizer.zero_grad()

                if self.model_type == "pairwise":
                    loss = self._train_step_pairwise(batch)
                else:
                    loss = self._train_step_e2rank(batch)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch Loss: {total_loss / len(loader):.4f}")

        print("\nTraining complete.")

    def _train_step_pairwise(self, batch):
        """Standard pairwise ranking loss"""
        queries, pos_docs, neg_docs = batch  # Unpack directly

        # Tokenize (Query, Pos) and (Query, Neg)
        pos_inputs = self.tokenizer(queries, pos_docs, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt").to(self.device)
        neg_inputs = self.tokenizer(queries, neg_docs, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt").to(self.device)

        pos_scores = self.model(**pos_inputs).logits.squeeze(-1)
        neg_scores = self.model(**neg_inputs).logits.squeeze(-1)

        return self.loss_fn(pos_scores, neg_scores)

    def _train_step_e2rank(self, batch):
        """E2Rank Layerwise loss"""
        queries, pos_docs, neg_docs = batch

        # Combine queries and docs for a single batch pass if possible,
        # but E2Rank usually treats Pos and Neg as simply "documents" to be classified against labels.
        # E2Rank loss expects model_outputs (with layer_logits) and labels.
        # We construct a batch of [ (Q, Pos), (Q, Neg) ]

        all_queries = list(queries) + list(queries)
        all_docs = list(pos_docs) + list(neg_docs)

        # Labels: 1 for Pos, 0 for Neg
        labels = torch.cat([
            torch.ones(len(queries), dtype=torch.long),
            torch.zeros(len(queries), dtype=torch.long)
        ]).to(self.device)

        inputs = self.tokenizer(all_queries, all_docs, padding=True,
                                truncation=True, max_length=512, return_tensors="pt").to(self.device)

        # Forward pass with hidden states
        outputs = self.model(**inputs, output_hidden_states=True)

        # Extract layer logits manually if model doesn't return them directly in the dict
        # (Our GroupedDebertaV2 returns them in .layer_logits if we used it, but let's be safe)
        if hasattr(outputs, "layer_logits"):
            layer_logits = outputs.layer_logits
        else:
            # Fallback: compute them if not present (though E2RankReranker's model should have them)
            # This part depends on GroupedDebertaV2 implementation
            layer_logits = []
            # Re-computing would be expensive.
            # Assuming GroupedDebertaV2 is used and works.
            # If standard model, this will fail.
            pass

        model_out = {
            "layer_logits": getattr(outputs, "layer_logits", []),
            "final_logits": outputs.logits
        }

        return self.loss_fn(model_out, labels)

    def save(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model_type == "pairwise":
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(),
                       output_dir / "pytorch_model.bin")
            self.tokenizer.save_pretrained(output_dir)

        print(f"Saved model to {output_dir}")