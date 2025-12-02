"""
Copyright:

  Copyright © 2025 uchuuronin 

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
  This file evaluates weight functions combined with reranker.
  Compares different weight combinations and strategies to find optimal config.

  Evaluations:
  1. Baseline comparisons (BM25 only, reranker only)
  2. Individual weight functions (consensus tfidf/dense, credibility)
     - Skip temporal for FEVER (no reliable date extraction)
  3. Combined weight functions with different strategies:
     - Neural LTR (learned combination)
     - Additive (simple average)
     - Harmonic (F-measure style, penalizes low scores)
  4. For each combination strategy, test both:
     - Consensus TF-IDF (CPU-friendly)
     - Consensus Dense (GPU, better quality)

Code:
IMPORTANT- YOU NEED TO GENERATE CACHED RERANKER RANK LIST (GPU REQUIRED)
THIS IS A COMPREHENSIVE EVALUATION
"""

from tqdm import tqdm
import argparse
import torch
import os
import json
from pyserini.search.lucene import LuceneSearcher
from src.config import (
    INDEX_DIR,
    DATA_DIR,
    TOP_QRELS_PATH,
    TOP_CLAIMS_PATH,
    TOP_RANKLISTS_PATH,
    RERANKEDLISTS_PATH,
)
from tests.utils import write_qrels, write_ranklists, eval_on_fever
from reranker import E2RankReranker
from reranker.weightfunc.consensus_weight import ConsensusWeightFunction
from reranker.weightfunc.credibility_weight import CredibilityWeightFunction
from reranker.weightfunc.temporal_weight import TemporalWeightFunction
from reranker.combine_weights import WeightCombiner

class WeightFunctionEvaluator:
    def __init__(self, max_claims=100, use_gpu=True):
        self.max_claims = max_claims
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_worker = 8
        
        print(f"Initializing WeightFunctionEvaluator...")
        print(f"GPU available: {torch.cuda.is_available()}")
        print(f"Using GPU: {self.use_gpu}")
        print(f"Max claims: {max_claims}")
        
        self._setup_data()
        self._setup_models()
    
    def _setup_data(self):
        print("\nSetting up data...")
        regenerate_ranklists = False
        regenerate_reranklists = False
        has_gpu = torch.cuda.is_available()

        if not (os.path.exists(TOP_QRELS_PATH) and \
                os.path.exists(TOP_CLAIMS_PATH)):
           print(f"Preparing {TOP_QRELS_PATH}...")
           print(f"Preparing {TOP_CLAIMS_PATH}...")
           regenerate_ranklists = True
           write_qrels(data_dir=DATA_DIR,
                       qrels_path=TOP_QRELS_PATH,
                       claims_path=TOP_CLAIMS_PATH,
                       max_claims=100)

        with open(TOP_QRELS_PATH, "r", encoding="utf8") as f:
            self.qrels = json.load(f)
        with open(TOP_CLAIMS_PATH, "r", encoding="utf8") as f:
            self.claims = json.load(f)
        
        self.searcher = LuceneSearcher(str(INDEX_DIR))
        self.searcher.set_bm25(1.2, 0.75)
        
        if (not os.path.exists(TOP_RANKLISTS_PATH)) or regenerate_ranklists:
            print(f"Preparing {TOP_RANKLISTS_PATH} (this will take awhile)...")
            write_ranklists(self.searcher,self.num_worker, TOP_RANKLISTS_PATH, self.claims, 50)

        with open(TOP_RANKLISTS_PATH, "r", encoding="utf8") as f:
            self.fever_ranklists = json.load(f)

        if (not os.path.exists(RERANKEDLISTS_PATH) or regenerate_reranklists) and has_gpu:
            print(f"Preparing {RERANKEDLISTS_PATH}  (this will take awhile)...")
            self.write_reranked_lists(self.claims, 10)

        if os.path.exists(RERANKEDLISTS_PATH):
            with open(RERANKEDLISTS_PATH, "r", encoding="utf8") as f:
                self.reranked_ranklists = json.load(f)
    
    def write_reranked_lists(self, raw_claims: list[dict], top_k: int) -> None:
        reranked_lists = {}
        
        print(f"\nReranking {len(raw_claims)} claims...")
        for claim_entry in tqdm(raw_claims, desc="Reranking", unit="claim"):
            claim_id = claim_entry['id']
            query = claim_entry['input']
            
            if claim_id not in self.fever_ranklists:
                continue
            
            bm25_docs = self.fever_ranklists[claim_id]
            
            doc_pairs = []
            for docid in bm25_docs.keys():
                doc = self.searcher.doc(docid)
                if doc is not None:
                    doc_text = doc.contents()
                    doc_pairs.append((docid, doc_text))
            
            if not doc_pairs:
                continue
            
            reranked = self.reranker.rerank(
                query=query,
                documents=doc_pairs,
                top_k=min(len(doc_pairs), top_k),
            )
            
            reranked_lists[claim_id] = {
                docid: float(score)
                for (docid, _text, score) in reranked
            }
        
        print(f"\nWriting {len(reranked_lists)} reranked results to {RERANKEDLISTS_PATH}...")
        with open(RERANKEDLISTS_PATH, "w", encoding="utf8") as f:
            json.dump(reranked_lists, f, indent=2)
    
    def _setup_models(self):
        print("\nSetting up models...")
        
        # Reranker
        if self.use_gpu:
            self.reranker = E2RankReranker(
                reranking_block_map={8: 50, 16: 28, 24: 10}
            )
            print(f"Loaded E2Rank reranker")
        else:
            self.reranker = None
            print(f"Skipping reranker (GPU required)")
        
        self.consensus_tfidf = ConsensusWeightFunction(
            min_similarity=0.15,
            multiplier=2.5,
            sim_method="tfidf"
        )
        print(f"Initialized Consensus (TF-IDF)")
        
        if self.use_gpu:
            self.consensus_dense = ConsensusWeightFunction(
                min_similarity=0.15,
                multiplier=3,
                sim_method="dense",
                device="cuda"
            )
            print(f"Initialized Consensus (Dense)")
        else:
            self.consensus_dense = None
        
        self.credibility = CredibilityWeightFunction(wikipedia_only=True)
        print(f"Initialized Credibility")
        self.temporal = TemporalWeightFunction(for_fever=True)
        print(f"Initialized Temporal")
        
    
    def _generate_weighted_ranklists(
        self,
        weight_functions: list,
        combination_method: str = "harmonic",
        model_path: str = None,
        top_k: int = 10
    ) -> dict:
        if combination_method == "neural":
            if model_path is None:
                model_path = "models/neural_combiner.pt"
            combiner = WeightCombiner(
                weight_functions=weight_functions,
                combination_method=combination_method,
                model_path=model_path
            )
        else:
            combiner = WeightCombiner(
                weight_functions=weight_functions,
                combination_method=combination_method
            )
        
        weighted_ranklists = {}
        
        for claim_entry in self.claims:
            claim_id = claim_entry['id']
            query = claim_entry['input']
            
            if claim_id not in self.reranked_ranklists:
                continue
            
            cached_scores = self.reranked_ranklists[claim_id]
            
            reranked = []
            for docid, score in cached_scores.items():
                doc = self.searcher.doc(docid)
                if doc is not None:
                    doc_text = doc.contents()
                    reranked.append((docid, doc_text, score))
            
            reranked.sort(key=lambda x: x[2], reverse=True)
            
            weighted = combiner.apply(query, reranked, alpha=0.5)  
            
            weighted_ranklists[claim_id] = {
                docid: float(score)
                for (docid, _text, score) in weighted[:top_k]
            }
            
            if weight_functions:
                raw_weights = []
                for wf in weight_functions:
                    w = wf.compute_weights(query, reranked)
                    raw_weights.append(w)
        
        return weighted_ranklists
    
    def evaluate_baseline_bm25(self):
        print("BASELINE: BM25 Only")
        
        bm25_top10 = {
            claim_id: dict(list(docs.items())[:10])
            for claim_id, docs in self.fever_ranklists.items()
        }
        
        metrics = eval_on_fever(self.qrels, bm25_top10, max_k=10)
        self._print_metrics(metrics)
        
        return metrics
    
    def evaluate_baseline_reranker(self):
        print("BASELINE: Reranker Only (no weights)")
        if not self.use_gpu:
            print("SKIPPED (GPU required)")
            return None
        
        ranklists = self._generate_weighted_ranklists(
            weight_functions=[],  # No weights
            top_k=10
        )
        
        metrics = eval_on_fever(self.qrels, ranklists, max_k=10)
        self._print_metrics(metrics)
        
        return metrics
    
    def evaluate_combined_strategy(
        self,
        consensus_method: str,
        combination_strategy: str,
        model_path: str = None
    ):
        consensus_fn = self.consensus_tfidf if consensus_method == "tfidf" else self.consensus_dense
        
        print(f"COMBINED: Consensus ({consensus_method.upper()}) + Credibility + Temporal ({combination_strategy.upper()})")
        
        if not self.use_gpu:
            print("SKIPPED (GPU required)")
            return None
        
        if consensus_fn is None:
            print(f"SKIPPED (Consensus {consensus_method} not available)")
            return None
        
        ranklists = self._generate_weighted_ranklists(
            weight_functions=[consensus_fn, self.credibility, self.temporal],
            combination_method=combination_strategy,
            model_path=model_path,
            top_k=10
        )
        
        metrics = eval_on_fever(self.qrels, ranklists, max_k=10)
        self._print_metrics(metrics)
        
        return metrics
    
    def _print_metrics(self, metrics: dict):
        metrics = {k: round(v, 3) for k, v in metrics.items()}
        print(metrics)
    
    def run_all_evaluations(self):
        print("RUNNING ALL EVALUATIONS")
        
        results = {}
        
        results["BM25 Only"] = self.evaluate_baseline_bm25()
        results["Reranker Only"] = self.evaluate_baseline_reranker()
        
        results["TF-IDF + Others (Harmonic)"] = self.evaluate_combined_strategy("tfidf", "harmonic")
        results["TF-IDF + Others (Additive)"] = self.evaluate_combined_strategy("tfidf", "additive")
        
        results["Dense + Others (Harmonic)"] = self.evaluate_combined_strategy("dense", "harmonic")
        results["Dense + Others (Additive)"] = self.evaluate_combined_strategy("dense", "additive")
        
        if os.path.exists("reranker/models/neural_combiner.pt"):
            results["Dense + Others (Neural LTR)"] = self.evaluate_combined_strategy(
                "dense", "neural", "reranker/models/neural_combiner.pt"
            )
            results["TF-IDF + Others (Neural LTR)"] = self.evaluate_combined_strategy(
                "tfidf", "neural", "reranker/models/neural_combiner.pt"
            )
        else:
            print("No LTR model found... Run ltr_train.py first to train the model")
        
        self._print_comparison_table(results)
        
        return results
    
    def _print_comparison_table(self, results: dict):
        print("COMPARISON TABLE")
        
        print(f"\n{'Configuration':<35} {'MAP@3':>7} {'MAP@5':>7} {'MAP@10':>8} {'P@3':>7} {'P@5':>7} {'P@10':>8} {'R@3':>7} {'R@5':>7} {'R@10':>8}")
        
        for config_name, metrics in results.items():
            if metrics is None:
                continue
            print(f"{config_name:<35} "
                f"{metrics.get('MAP_3', 0):>7.3f} "
                f"{metrics.get('MAP_5', 0):>7.3f} "
                f"{metrics.get('MAP_10', 0):>8.3f} "
                f"{metrics.get('P_3', 0):>7.3f} "
                f"{metrics.get('P_5', 0):>7.3f} "
                f"{metrics.get('P_10', 0):>8.3f} "
                f"{metrics.get('R_3', 0):>7.3f} "
                f"{metrics.get('R_5', 0):>7.3f} "
                f"{metrics.get('R_10', 0):>8.3f}")
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_config = max(valid_results.items(), key=lambda x: x[1]['MAP_10'])
            print(f"Best performing model (Based on MAP): {best_config}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate weight functions for fact verification")
    parser.add_argument("--max-claims", type=int, default=100,
                       help="Number of claims to evaluate (default: 100)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU usage (will skip reranker tests)")
    args = parser.parse_args()
    
    evaluator = WeightFunctionEvaluator(
        max_claims=args.max_claims,
        use_gpu=not args.no_gpu
    )
    
    results = evaluator.run_all_evaluations()
    

if __name__ == "__main__":
    main()