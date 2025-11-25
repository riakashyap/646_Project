"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file tests the efficacy of the BM25 retrieval stage by using mean
  average precision.

  General approach adapted from materials presented in the course curriculum
  COMPSCI646, UMass Amherst, Fall 2025.

Code:
"""

from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict
from src.config import (
    INDEX_DIR,
    QRELS_PATH,
    DATA_DIR,
    CLAIMS_PATH,
    RANKLISTS_PATH
)
from tests.utils import write_qrels, write_ranklists
import os
import json
import unittest
from datasets import load_dataset
import pytrec_eval


class TestBM25(unittest.TestCase):

    searcher: LuceneSearcher
    num_worker: int = 8
    qrels: dict[str, dict[str, int]]
    fever_ranklists: dict[str, dict[str, float]]

    @classmethod
    def setUpClass(self):
        super().setUpClass()

        regenerate_ranklists = False

        if not (os.path.exists(QRELS_PATH) and \
                os.path.exists(CLAIMS_PATH)):
           print(f"Preparing {QRELS_PATH}...")
           print(f"Preparing {CLAIMS_PATH}...")
           regenerate_ranklists = True
           write_qrels(data_dir=DATA_DIR,
                       qrels_path=QRELS_PATH,
                       claims_path=CLAIMS_PATH)

        with open(QRELS_PATH, "r", encoding="utf8") as f:
            self.qrels = json.load(f)
        with open(CLAIMS_PATH, "r", encoding="utf8") as f:
            claims = json.load(f)

        self.searcher = LuceneSearcher(str(INDEX_DIR))
        self.searcher.set_bm25(1.2, 0.75)

        if (not os.path.exists(RANKLISTS_PATH)) or regenerate_ranklists:
            print(f"Preparing {RANKLISTS_PATH} (this will take awhile)...")
            write_ranklists(self.searcher,self.num_worker, RANKLISTS_PATH, claims, 10)

        with open(RANKLISTS_PATH, "r", encoding="utf8") as f:
            self.fever_ranklists = json.load(f)

    def test_can_retrieve(self):
        query = "serval"
        hits = self.searcher.search(query, k=20)

        docids = [hit.docid for hit in hits]
        docs = [self.searcher.doc(docid) for docid in docids]
        self.assertGreater(len(hits), 0, "BM25 search returned no documents.")

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

        expected = {
            "P_3": 0.124,
            "P_5": 0.090,
            "P_10": 0.056,
            "R_3": 0.334,
            "R_5": 0.403,
            "R_10": 0.498,
            "MAP_3": 0.257,
            "MAP_5": 0.273,
            "MAP_10": 0.286,
        }
        actual = eval_on_fever()
        actual = {key: round(value, 3) for key, value in actual.items()}
        self.assertEqual(expected, actual, "BM25 evaluation on the fever dataset"
                         " was significantly different than expected!")
