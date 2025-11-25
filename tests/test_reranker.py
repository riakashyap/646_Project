"""
Copyright:

  Copyright © 2025 uchuuronin 

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file tests the efficacy of the reranker post BM25 retrieval stage by 
  using mean average precision (MAP)

  General approach adapted from materials presented in the course curriculum
  COMPSCI646, UMass Amherst, Fall 2025.

Code:
"""
import torch
from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict
from src.config import (
    INDEX_DIR,
    DATA_DIR,
    TOP_QRELS_PATH,
    TOP_CLAIMS_PATH,
    TOP_RANKLISTS_PATH,
    RERANKEDLISTS_PATH
)
from tests.utils import write_qrels, write_ranklists, eval_on_fever
import os
import json
import unittest
from datasets import load_dataset
import pytrec_eval
import random
from reranker import E2RankReranker

class TestReranker(unittest.TestCase):

    searcher: LuceneSearcher
    num_worker: int = 8
    qrels: dict[str, dict[str, int]]
    bm25_ranklists_top50: dict[str, dict[str, float]]
    reranked_ranklists: dict[str, dict[str, float]]
    reranker = E2RankReranker(
        reranking_block_map={8: 50, 16: 28, 24: 10}
    )

    @classmethod
    def setUpClass(self):
        super().setUpClass()
        
        regenerate_ranklists = False
        regenerate_reranklists = False

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
            claims = json.load(f)
        
        self.searcher = LuceneSearcher(str(INDEX_DIR))
        self.searcher.set_bm25(1.2, 0.75)
        
        if (not os.path.exists(TOP_RANKLISTS_PATH)) or regenerate_ranklists:
            print(f"Preparing {TOP_RANKLISTS_PATH} (this will take awhile)...")
            write_ranklists(self.searcher,self.num_worker, TOP_RANKLISTS_PATH, claims, 50)

        with open(TOP_RANKLISTS_PATH, "r", encoding="utf8") as f:
            self.fever_ranklists = json.load(f)

        if not os.path.exists(RERANKEDLISTS_PATH) or regenerate_reranklists:
            print(f"Preparing {RERANKEDLISTS_PATH}  (this will take awhile)...")
            self.write_reranked_lists(claims, 10)

        with open(RERANKEDLISTS_PATH, "r", encoding="utf8") as f:
            self.reranked_ranklists = json.load(f)

    def test_fever_evaluation(self):
        expected = {
            "P_3": 0.107,
            "P_5": 0.08,
            "P_10": 0.051,
            "P_50": 0.017,
            "R_3": 0.31,
            "R_5": 0.39,
            "R_10": 0.475,
            "R_50": 0.747,
            "MAP_3": 0.252 ,
            "MAP_5": 0.271,
            "MAP_10": 0.283,
            "MAP_50": 0.296,
        }
        actual = eval_on_fever(self.qrels, self.fever_ranklists, max_k=50)
        actual = {key: round(value, 3) for key, value in actual.items()}
        self.assertEqual(expected, actual, "BM25 evaluation on the fever dataset"
                         " was significantly different than expected!")

    def test_can_rerank(self):
        claim_id = list(self.reranked_ranklists.keys())[0]
        reranked_docs = self.reranked_ranklists[claim_id]
        
        self.assertGreater(len(reranked_docs), 0, 
                          "Reranker returned no documents.")
        self.assertLessEqual(len(reranked_docs), 10,
                            "Reranker returned more than top-10 docs")
        
    def test_reranker_evaluation(self):
        rerank_metrics = eval_on_fever(self.qrels, self.reranked_ranklists, max_k=50)
        rerank_metrics = {k: round(v, 3) for k, v in rerank_metrics.items()}

        print("Reranker metrics:", rerank_metrics)
        # TODO: add assertions based on expected performance
                 
    @classmethod
    def write_reranked_lists(self,
                          raw_claims: list[dict],
                          top_k: int) -> None:
        reranked_lists: dict[str, dict[str, float]] = {}
        with open(TOP_RANKLISTS_PATH, "r", encoding="utf8") as f:
            bm25_ranklists = json.load(f)
            
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
        for claim_entry in raw_claims: 
            claim_id = claim_entry['id']
            query = claim_entry['input']
            
            if claim_id not in bm25_ranklists:
                continue
                
            bm25_docs = bm25_ranklists[claim_id]
            
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
        with open(RERANKEDLISTS_PATH, "w", encoding="utf8") as out:
            json.dump(reranked_lists, out, indent=2)
        
