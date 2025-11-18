#!/usr/bin/env python3
"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file tests the effectiveness of the reranker by comparing BM25 alone
  vs BM25 + Reranker. It outputs comprehensive metrics and logs.

Code:
"""

import unittest
import sys
import os
from pathlib import Path
from typing import List, Tuple
import time
from collections import defaultdict
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import load_dataset, Dataset, concatenate_datasets
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reranker import E2RankReranker
from src.config import INDEX_DIR, DATA_DIR
from src.model_clients import LlamaCppClient
from src.ragar_corag import RagarCorag
from src.config import PROMPTS_DIR

class TestReranker(unittest.TestCase):
    """
    Comprehensive test comparing BM25 baseline vs BM25 + Reranker.
    
    Outputs:
    - Accuracy, Precision, Recall, F1 for each configuration
    - Initial retreived list of results vs results post-reranking vs true expected results
    - Number of "NOT ENOUGH INFO"//NEI predictions
    - Time taken per claim
    - Detailed log file
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        super().setUpClass()
        
        # Modify these parameters to test different configurations
        cls.num_claims = 50
        cls.think_mode = False
        cls.use_ragar_prompts = True
        cls.model_size = "8b" # Qwen 
        
        # Determine prompt directories
        if cls.use_ragar_prompts:
            cls.user_prompts_dir = PROMPTS_DIR / "ragar"
            cls.sys_prompts_dir = None
            cls.prompt_type = "r"
        else:
            cls.user_prompts_dir = PROMPTS_DIR / "custom" / "user"
            cls.sys_prompts_dir = PROMPTS_DIR / "custom" / "system"
            cls.prompt_type = ""
        
        cls.mc = LlamaCppClient(
            cls.user_prompts_dir,
            cls.sys_prompts_dir,
            think_mode_bool=cls.think_mode
        )
        
        print("\n[Loading FEVER dataset...")
        ds = load_dataset("fever", "v1.0", trust_remote_code=True)
        split = ds["train"]
        split = Dataset.from_pandas(split.to_pandas().drop_duplicates(subset="claim"))
        
        # Get balanced samples
        supports = split.filter(lambda row: row["label"] == "SUPPORTS").select(range(cls.num_claims // 2))
        refutes = split.filter(lambda row: row["label"] == "REFUTES").select(range(cls.num_claims // 2))
        cls.test_data = concatenate_datasets([supports, refutes])
        
        print(f"Testing configurations:")
        print(f"\tClaims: {cls.num_claims}")
        print(f"\tThink mode: {cls.think_mode}")
        print(f"\tPrompts: {'RAGAR' if cls.use_ragar_prompts else 'CUSTOM'}")
        print(f"\tModel: Qwen3 {cls.model_size}")
    
    
    def _run_pipeline(self, use_reranker: bool) -> Tuple[List[str], List[str], List[str], float]:
        """
        Returns:
            (bm25_labels, predicted_labels, true_labels, elapsed_time)
        """
        reranker = None
        if use_reranker:
            print("\nInitializing E2Rank reranker...")
            reranker = E2RankReranker()
            print(f"Reranker loaded successfully")
        
        corag = RagarCorag(self.mc, reranker=reranker)
        
        fever_labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
        predicted_labels = []
        true_labels = []
        
        start_time = time.time()
        
        config_name = "WITH reranker" if use_reranker else "WITHOUT reranker"
        print(f"\nRunning pipeline {config_name}...")
        
        for i in tqdm(range(len(self.test_data)), desc="Processing claims"):
            claim = self.test_data[i]["claim"]
            label = self.test_data[i]["label"]
            
            try:
                result = corag.run(claim)
                verdict = result["verdict"]
                pred = None if verdict is None else fever_labels[verdict]
            except Exception as e:
                print(f"\nError processing claim {i}: {e}")
                pred = "NOT ENOUGH INFO"
            
            predicted_labels.append(pred)
            true_labels.append(label)
        
        elapsed_time = time.time() - start_time
        
        return predicted_labels, true_labels, elapsed_time
    
    
    def _compute_metrics(self, predicted_labels: List[str], true_labels: List[str]) -> dict:
        """
        Compute accuracy, precision, recall, F1, and confusion matrix.
        
        Returns:
            Dictionary with all metrics
        """
        # Handle None predictions as "NOT ENOUGH INFO"
        predicted_labels = [pred if pred is not None else "NOT ENOUGH INFO" for pred in predicted_labels]
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels,
            labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
            average=None,
            zero_division=0
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels,
            average='weighted',
            zero_division=0
        )
        
        labels_order = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)
        
        nei_count = predicted_labels.count("NOT ENOUGH INFO")
        
        metrics = {
            "accuracy": accuracy,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "per_class": {
                "SUPPORTS": {
                    "precision": precision[0],
                    "recall": recall[0],
                    "f1": f1[0],
                    "support": int(support[0])
                },
                "REFUTES": {
                    "precision": precision[1],
                    "recall": recall[1],
                    "f1": f1[1],
                    "support": int(support[1])
                }
            },
            "confusion_matrix": cm.tolist(),
            "nei_predictions": nei_count
        }
        
        return metrics
    
    def _write_log(self, config_name: str, metrics: dict, elapsed_time: float,
               predicted_labels: List[str], true_labels: List[str]):
        """Write comprehensive log file."""
        
        logs_dir = Path(__file__).parent.parent / "logs/reranker"
        logs_dir.mkdir(exist_ok=True, parents=True)
        
        thinking_flag = "t-" if self.think_mode else ""
        prompt_flag = "r-" if self.use_ragar_prompts else ""
        reranker_flag = "reranker" if "reranker" in config_name else ""
        
        parts = [prompt_flag, thinking_flag, self.model_size, reranker_flag]
        filename_parts = [p.rstrip('-') for p in parts if p]
        filename = f"log-{'-'.join(filename_parts)}.txt"
        filepath = logs_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"Configuration: {config_name}\n")
            f.write(f"Think mode: {self.think_mode}\n")
            f.write(f"Prompts: {'RAGAR' if self.use_ragar_prompts else 'CUSTOM'}\n")
            f.write(f"Model: Qwen3 {self.model_size}\n")
            f.write(f"Number of claims: {len(true_labels)}\n")
            f.write("\n")
            
            f.write(f"Accuracy: {metrics['accuracy']:.8f}\n")
            f.write(f"Weighted Precision: {metrics['weighted_precision']:.8f}\n")
            f.write(f"Weighted Recall: {metrics['weighted_recall']:.8f}\n")
            f.write(f"Weighted F1: {metrics['weighted_f1']:.8f}\n")
            f.write("\n")
            
            f.write("Per-Class Metrics:\n")
            for label in ["SUPPORTS", "REFUTES"]:
                metrics_for_label = metrics['per_class'][label]
                f.write(f"{label}:\n")
                f.write(f"\tPrecision: {metrics_for_label['precision']:.4f}\n")
                f.write(f"\tRecall: {metrics_for_label['recall']:.4f}\n")
                f.write(f"\tF1: {metrics_for_label['f1']:.4f}\n")
                f.write(f"\tActual: {metrics_for_label['support']}\n")
                predicted_count = predicted_labels.count(label)
                f.write(f"\tPredicted: {predicted_count}\n")
            f.write("\n")
            
            f.write(f"NOT ENOUGH INFO predictions: {metrics['nei_predictions']}\n")
            f.write("\n")
            
            f.write(f"Time: {elapsed_time:.6f} seconds\n")
            f.write(f"Time per claim: {elapsed_time / len(true_labels):.6f} seconds\n")
            f.write("\n")
            
            f.write(f"Predicted labels: {predicted_labels}\n")
            f.write(f"True labels: {true_labels}\n")
        
        print(f"\nDoneeee, the log has been written to: {filepath}")
    
    def test_bm25_baseline(self):
        """Test BM25 retrieval without reranker (k=3 direct)."""
        print("BM25 Baseline (No Reranker)")
        
        predicted_labels, true_labels, elapsed_time = self._run_pipeline(use_reranker=False)
        metrics = self._compute_metrics(predicted_labels, true_labels)
        
        print(f"\n\tAccuracy:\t{metrics['accuracy']:.4f}")
        print(f"\tWeighted F1:\t{metrics['weighted_f1']:.4f}")
        print(f"\tNEI Predicted:\t{metrics['nei_predictions']}")
        print(f"\tTotal Time:\t{elapsed_time:.2f}s")
        print(f"\tTime/Claim:\t{elapsed_time/len(true_labels):.3f}s")
        
        self._write_log("baseline", metrics, elapsed_time, predicted_labels, true_labels)

    def test_bm25_with_reranker(self):
        """Test BM25 retrieval with reranker"""
        print("BM25 + E2Rank Reranker")
        
        predicted_labels, true_labels, elapsed_time = self._run_pipeline(use_reranker=True)
        metrics = self._compute_metrics(predicted_labels, true_labels)
        
        print(f"\n\tAccuracy:\t{metrics['accuracy']:.4f}")
        print(f"\tWeighted F1:\t{metrics['weighted_f1']:.4f}")
        print(f"\tNEI Predicted:\t{metrics['nei_predictions']}")
        print(f"\tTotal Time:\t{elapsed_time:.2f}s")
        print(f"\tTime/Claim:\t{elapsed_time/len(true_labels):.3f}s")
        
        self._write_log("with_reranker", metrics, elapsed_time, predicted_labels, true_labels)

if __name__ == "__main__":
    unittest.main(verbosity=2)
