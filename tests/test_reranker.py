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

HAS_GPU = torch.cuda.is_available()

@unittest.skipIf(not HAS_GPU, "Reranker tests require GPU, wwhich is not enabled. Skipping unittest...")
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

        if (not os.path.exists(RERANKEDLISTS_PATH) or regenerate_reranklists):
            print(f"Preparing {RERANKEDLISTS_PATH}  (this will take awhile)...")
            self.write_reranked_lists(claims, 10)

        if os.path.exists(RERANKEDLISTS_PATH):
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

    @unittest.skipIf(not os.path.exists(RERANKEDLISTS_PATH), "Reranked lists not generated. Run with GPU to generate.")
    def test_can_rerank(self):
        claim_id = list(self.reranked_ranklists.keys())[0]
        reranked_docs = self.reranked_ranklists[claim_id]
        
        self.assertGreater(len(reranked_docs), 0, 
                          "Reranker returned no documents.")
        self.assertLessEqual(len(reranked_docs), 10,
                            "Reranker returned more than top-10 docs")
    
    @unittest.skipIf(not os.path.exists(RERANKEDLISTS_PATH), "Reranked lists not generated. Run with GPU to generate.")
    def test_reranker_evaluation(self):
        rerank_metrics = eval_on_fever(self.qrels, self.reranked_ranklists, max_k=50)
        rerank_metrics = {k: round(v, 3) for k, v in rerank_metrics.items()}
        bm25_metrics = eval_on_fever(self.qrels, self.fever_ranklists, max_k=50)
        bm25_metrics = {k: round(v, 3) for k, v in bm25_metrics.items()}


        self.assertTrue(
            all(0 <= v <= 1 for v in rerank_metrics.values()),
            f"All metrics not in valid range [0,1]. Got: {rerank_metrics}"
        )
        self.assertGreaterEqual(
            rerank_metrics['MAP_10'], 
            bm25_metrics['MAP_10'],
            f"Reranker underperformed BM25 on MAP_10"
        )
        
    def test_compute_score(self):
        query = "Who is the president of the United States?"
        doc = "Joe Biden is the current president of the United States."
        
        scores = [round(self.reranker.compute_score(query, doc), 5) for _ in range(5)]
        unique_scores = len(set(scores))
        self.assertEqual(
            unique_scores, 1,
            f"In five calls, got {unique_scores} unique score: {scores}. Model appears to not be consistent in scoring."
        )

    def test_batch_scores(self):
        query = "What is the capital of France?"
        docs = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "The weather today is sunny."
        ]
        
        runs = [
            [round(s, 5) for s in self.reranker.batch_compute_scores(query, docs)]
            for _ in range(5)
        ]
        
        non_deterministic_docs = [] # records doc if scores varied for that doc across runs
        for doc_idx in range(len(docs)):
            doc_scores = [run[doc_idx] for run in runs]
            unique = len(set(doc_scores))
            if unique > 1:
                non_deterministic_docs.append((doc_idx, unique))
                
        self.assertEqual(
            unique, 1,
            f"Model is not consistent in scoring across batch runs for all docs."
        ) 

    def test_rerank(self):
        query = "Who invented the telephone?"
        doc_pairs = [
            ("doc1", "Alexander Graham Bell invented the telephone."),
            ("doc2", "Thomas Edison invented the light bulb."),
            ("doc3", "The telephone was invented in 1876."),
        ]
        
        orderings = []
        score_sets = []
        for _ in range(5):
            reranked = self.reranker.rerank(query, doc_pairs, top_k=3)
            order = tuple(doc_id for doc_id, _, _ in reranked)
            scores = tuple(round(score, 5) for _, _, score in reranked)
            orderings.append(order)
            score_sets.append(scores)
        
        unique_orderings = len(set(orderings))
        unique_score_sets = len(set(score_sets)) 
        self.assertTrue(
            all(len(order) == 3 for order in orderings),
            f"Rankings failed to returne 3 docs, not following top_k parameter."
        )
        self.assertEqual( 
            unique_orderings, 1, 
            f"{unique_orderings} ordering variations for docs indicating model is not consistent." 
        )

    @classmethod
    def write_reranked_lists(self,
                          raw_claims: list[dict],
                          top_k: int) -> None:
        reranked_lists: dict[str, dict[str, float]] = {}
        with open(TOP_RANKLISTS_PATH, "r", encoding="utf8") as f:
            bm25_ranklists = json.load(f)
            
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

        with open(RERANKEDLISTS_PATH, "w", encoding="utf8") as out:
            json.dump(reranked_lists, out, indent=2)