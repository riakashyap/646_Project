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
from tests.utils import write_qrels, write_ranklists, eval_on_fever
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
        actual = eval_on_fever(self.qrels, self.fever_ranklists)
        actual = {key: round(value, 3) for key, value in actual.items()}
        self.assertEqual(expected, actual, "BM25 evaluation on the fever dataset"
                         " was significantly different than expected!")
