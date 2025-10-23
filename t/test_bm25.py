"""
Copyright:

  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file tests the efficacy of the BM25 retrieval stage by using mean
  average precision.

  Some of this was adapted from materials presented in the course curriculum
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
           self.write_qrels()

        with open(QRELS_PATH, "r", encoding="utf8") as f:
            self.qrels = json.load(f)
        with open(CLAIMS_PATH, "r", encoding="utf8") as f:
            claims = json.load(f)

        self.searcher = LuceneSearcher(str(INDEX_DIR))
        self.searcher.set_bm25(1.2, 0.75)

        if (not os.path.exists(RANKLISTS_PATH)) or regenerate_ranklists:
            print(f"Preparing {RANKLISTS_PATH} (this will take awhile)...")
            self.write_ranklists(claims, 10)

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

    @classmethod
    def write_ranklists(self,
                     raw_claims: list[dict],
                     top_k: int) -> None:
        """
        Runs BM25 retrieval with Pyserini for a subset of fever claims.
        """
        ranklists: dict[str, dict[str, float]] = {}

        list_claims = []
        list_ids = []
        for entry in raw_claims:
            list_ids.append(entry['id'])
            list_claims.append(entry['input'])

        hits = self.searcher.batch_search(
            list_claims,
            qids=list_ids,
            k=top_k,
            threads=self.num_worker
        )

        for claim_id, curr_q_hits in hits.items():
            retrieved_docs: dict[str, float] = {}
            for h in curr_q_hits:
                retrieved_docs[h.docid] = float(h.score)
                ranklists[claim_id] = retrieved_docs

        with open(RANKLISTS_PATH, "w", encoding="utf8") as out:
            json.dump(ranklists, out, indent=2)

    @classmethod
    def write_qrels(self) -> None:
        """
        Downloads and generates a QRELS, CLAIMS file out of the fever training set.
        """
        os.makedirs(DATA_DIR, exist_ok=True)

        ds = load_dataset(
            "fever",
            "v1.0",
            cache_dir=DATA_DIR,
            split="train"
        )

        qrels = defaultdict(lambda: defaultdict(lambda: 0))
        claims = []
        added_claims = set()

        for ex in ds:
            cid = str(ex["id"])
            l = ex["label"]
            if l not in ("SUPPORTS", "REFUTES"):
                # don't care if there's (possibly no) evidence
                continue
            page = ex.get("evidence_wiki_url")
            sent_id = ex.get("evidence_sentence_id")
            claim = ex.get("claim")
            if (page is not None and
                sent_id is not None and
                claim is not None
                ):
                qrels[cid][page] = 1

                # avoid adding the same cid twice
                if cid not in added_claims:
                    claims.append({
                        "id": cid,
                        "input": claim,
                    })
                added_claims.add(cid)

        with open(QRELS_PATH, "w", encoding="utf8") as out:
            json.dump(qrels, out, indent=2)
        with open(CLAIMS_PATH, "w", encoding="utf8") as out:
            json.dump(claims, out, indent=2)
