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
    QRELS_PATH,
    DATA_DIR,
    CLAIMS_PATH,
    RANKLISTS_PATH,
    RERANKEDLISTS_PATH
)
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
        reranking_block_map={
            8: 50,
            16: 20,
            24: 10,
        }
    )

    @classmethod
    def setUpClass(self):
        super().setUpClass()

        regenerate_reranklists = False

        with open(QRELS_PATH, "r", encoding="utf8") as f:
            self.qrels = json.load(f)
        with open(CLAIMS_PATH, "r", encoding="utf8") as f:
            claims = json.load(f)

        self.searcher = LuceneSearcher(str(INDEX_DIR))
        self.searcher.set_bm25(1.2, 0.75)

        with open(RANKLISTS_PATH, "r", encoding="utf8") as f:
            self.fever_ranklists = json.load(f)
            
        if not os.path.exists(RERANKEDLISTS_PATH) or regenerate_reranklists:
            print(f"Preparing {RERANKEDLISTS_PATH}  (this will take awhile)...")
            self.write_reranked_lists(claims, 10)

        with open(RERANKEDLISTS_PATH, "r", encoding="utf8") as f:
            self.reranked_ranklists = json.load(f)

    def test_fever_evaluation(self):
        def eval_on_fever() -> dict[str, float]:
            """
            Evaluate BM25 retrieval results using pytrec_eval.
            """

            evaluator = pytrec_eval.RelevanceEvaluator(
                self.qrels,
                {
                    'P.3', 'P.5', 'P.10',
                    'recall.3', 'recall.5', 'recall.10',
                    'map_cut.3', 'map_cut.5', 'map_cut.10'
                }
            )
            results = evaluator.evaluate(self.fever_ranklists)

            P_3, P_5, P_10 = 0, 0, 0
            R_3, R_5, R_10 = 0, 0, 0
            MAP_3, MAP_5, MAP_10 = 0, 0, 0

            for query_id, scores in results.items():
                P_3 += scores['P_3']
                P_5 += scores['P_5']
                P_10 += scores['P_10']
                R_3 += scores['recall_3']
                R_5 += scores['recall_5']
                R_10 += scores['recall_10']
                MAP_3 += scores['map_cut_3']
                MAP_5 += scores['map_cut_5']
                MAP_10 += scores['map_cut_10']

            num_queries = len(self.fever_ranklists)

            return {
                "P_3": P_3 / num_queries,
                "P_5": P_5 / num_queries,
                "P_10": P_10 / num_queries,
                "R_3": R_3 / num_queries,
                "R_5": R_5 / num_queries,
                "R_10": R_10 / num_queries,
                "MAP_3": MAP_3 / num_queries,
                "MAP_5": MAP_5 / num_queries,
                "MAP_10": MAP_10 / num_queries,
            }

        actual = eval_on_fever()
        actual = {key: round(value, 3) for key, value in actual.items()}
        print("\nBM25 metrics:", actual)

    def test_can_rerank(self):
        claim_id = list(self.reranked_ranklists.keys())[0]
        reranked_docs = self.reranked_ranklists[claim_id]
        
        self.assertGreater(len(reranked_docs), 0, 
                          "Reranker returned no documents.")
        self.assertLessEqual(len(reranked_docs), 10,
                            "Reranker returned more than top-10 docs")
        
    def test_reranker_evaluation(self):
        def eval_on_fever() -> dict[str, float]:
            """
            Evaluate RERANKED retrieval results using pytrec_eval.
            """
            evaluator = pytrec_eval.RelevanceEvaluator(
                self.qrels,
                {
                    'P.3', 'P.5', 'P.10',
                    'recall.3', 'recall.5', 'recall.10',
                    'map_cut.3', 'map_cut.5', 'map_cut.10'
                }
            )
            results = evaluator.evaluate(self.reranked_ranklists)

            P_3 = P_5 = P_10 = 0.0
            R_3 = R_5 = R_10 = 0.0
            MAP_3 = MAP_5 = MAP_10 = 0.0

            for _qid, scores in results.items():
                P_3 += scores["P_3"]
                P_5 += scores["P_5"]
                P_10 += scores["P_10"]
                R_3 += scores["recall_3"]
                R_5 += scores["recall_5"]
                R_10 += scores["recall_10"]
                MAP_3 += scores["map_cut_3"]
                MAP_5 += scores["map_cut_5"]
                MAP_10 += scores["map_cut_10"]

            num_queries = len(self.reranked_ranklists) if self.reranked_ranklists else 1

            return {
                "P_3": P_3 / num_queries,
                "P_5": P_5 / num_queries,
                "P_10": P_10 / num_queries,
                "R_3": R_3 / num_queries,
                "R_5": R_5 / num_queries,
                "R_10": R_10 / num_queries,
                "MAP_3": MAP_3 / num_queries,
                "MAP_5": MAP_5 / num_queries,
                "MAP_10": MAP_10 / num_queries,
            }

        rerank_metrics = eval_on_fever()
        rerank_metrics = {k: round(v, 3) for k, v in rerank_metrics.items()}

        print("Reranker metrics:", rerank_metrics)

    @classmethod
    def write_reranked_lists(self,
                          raw_claims: list[dict],
                          top_k: int) -> None:
        reranked_lists: dict[str, dict[str, float]] = {}
        with open(RANKLISTS_PATH, "r", encoding="utf8") as f:
            bm25_ranklists = json.load(f)
            
        print(f"\nReranking {len(raw_claims)} claims...")
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

        with open(RERANKEDLISTS_PATH, "w", encoding="utf8") as out:
            json.dump(reranked_lists, out, indent=2)
        
