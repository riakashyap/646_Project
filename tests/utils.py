"""
Copyright:
  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
  This file contains common functions called in multiple test scripts.
  
Code:
"""

from typing import Dict, List, Tuple, Optional
from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict
from datasets import load_dataset
import json
import os
import pytrec_eval

def write_qrels(
    qrels_path: str,
    claims_path: str,
    data_dir: str,
    max_claims: Optional[int] = None
) -> None:
    """
    Downloads and generates a QRELS, CLAIMS file out of the fever training set.
    """
    os.makedirs(data_dir, exist_ok=True)

    ds = load_dataset(
        "fever",
        "v1.0",
        cache_dir=data_dir,
        split="train"
    )

    qrels = defaultdict(lambda: defaultdict(lambda: 0))
    claims = []
    added_claims = set()
    
    for ex in ds:
        if max_claims is not None and len(added_claims) >= max_claims:
            break
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

    with open(qrels_path, "w", encoding="utf8") as out:
        json.dump(qrels, out, indent=2)
    with open(claims_path, "w", encoding="utf8") as out:
        json.dump(claims, out, indent=2)

def write_ranklists(
    lucene_searcher: LuceneSearcher,
    num_worker: int,
    ranklist_path: str,
    raw_claims: list[dict],
    top_k: int
    ) -> None:
    """
    Runs BM25 retrieval with Pyserini for a subset of fever claims.
    """
    ranklists: dict[str, dict[str, float]] = {}

    list_claims = []
    list_ids = []
    for entry in raw_claims:
        list_ids.append(entry['id'])
        list_claims.append(entry['input'])

    hits = lucene_searcher.batch_search(
        list_claims,
        qids=list_ids,
        k=top_k,
        threads= num_worker
    )

    for claim_id, curr_q_hits in hits.items():
        retrieved_docs: dict[str, float] = {}
        for h in curr_q_hits:
            retrieved_docs[h.docid] = float(h.score)
        ranklists[claim_id] = retrieved_docs

    with open(ranklist_path, "w", encoding="utf8") as out:
        json.dump(ranklists, out, indent=2)
        
def eval_on_fever(
    qrels: Dict[str, Dict[str, int]],
    ranklists: Dict[str, Dict[str, float]],
    max_k: int = 10
    ) -> dict[str, float]:
    """
    Evaluate BM25 retrieval results using pytrec_eval.
    """
    eval_checkpoints = [3, 5, 10]
    
    k_values = sorted(set([k for k in eval_checkpoints if k <= max_k] + [max_k]))
    # if max_k >10, then include max_k in evaluation
    
    metrics = set()
    for k in k_values:
        metrics.add(f'P.{k}')
        metrics.add(f'recall.{k}')
        metrics.add(f'map_cut.{k}')
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(ranklists)

    P_dict = {k: 0.0 for k in k_values}
    R_dict = {k: 0.0 for k in k_values}
    MAP_dict = {k: 0.0 for k in k_values}

    for query_id, scores in results.items():
        for k in k_values:
            P_dict[k] += scores[f'P_{k}']
            R_dict[k] += scores[f'recall_{k}']
            MAP_dict[k] += scores[f'map_cut_{k}']

    num_queries = len(ranklists)

    result = {}
    for k in k_values:
        result[f"P_{k}"] = P_dict[k] / num_queries
        result[f"R_{k}"] = R_dict[k] / num_queries
        result[f"MAP_{k}"] = MAP_dict[k] / num_queries
    
    return result