"""
BM25 Parameter Optimization Script

Grid searches BM25 (k1, b) and optional RM3 query expansion on a subset of FEVER
claims, computing Precision, Recall, MAP, and MADR at cutoffs 3,5,10.

Outputs a ranked summary of parameter sets and recommends the best.

Usage:
  python scripts/optimize_bm25.py --sample-size 5000 --rm3 --fb-docs 10 --fb-terms 10 --orig-weight 0.7

Requires existing FEVER qrels and claims JSON produced by tests/test_bm25.py.
If missing, run:  python -m tests.test_bm25  (or run_tests.py)
"""
from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List
from collections import defaultdict
from statistics import mean

import sys
from pathlib import Path

# Ensure project root is on sys.path when script is executed directly
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import QRELS_PATH, CLAIMS_PATH, INDEX_DIR
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

@dataclass
class Metrics:
    P_3: float
    P_5: float
    P_10: float
    R_3: float
    R_5: float
    R_10: float
    MAP_3: float
    MAP_5: float
    MAP_10: float
    MADR_3: float
    MADR_5: float
    MADR_10: float

    def score(self, objective: str) -> float:
        """Objective scoring function.

        objective choices:
          - 'composite' (default): 0.6MAP@10 + 0.4(1 - MADR@10/10)
          - 'map10': MAP@10
          - 'p3' / 'p5' / 'p10': Precision at k
          - 'madr3' / 'madr5' / 'madr10': negative MADR (lower is better)
        """
        if objective == 'map10':
            return self.MAP_10
        if objective == 'p3':
            return self.P_3
        if objective == 'p5':
            return self.P_5
        if objective == 'p10':
            return self.P_10
        if objective == 'madr3':
            return -self.MADR_3
        if objective == 'madr5':
            return -self.MADR_5
        if objective == 'madr10':
            return -self.MADR_10

        # composite
        inv_madr10 = (10 - min(self.MADR_10, 10)) / 10
        return 0.6 * self.MAP_10 + 0.4 * inv_madr10

@dataclass
class ParamSet:
    k1: float
    b: float
    rm3: bool
    fb_docs: int | None = None
    fb_terms: int | None = None
    orig_weight: float | None = None

    def label(self) -> str:
        if not self.rm3:
            return f"k1={self.k1:.2f}, b={self.b:.2f}" \
                + " (bm25)"
        return (
            f"k1={self.k1:.2f}, b={self.b:.2f} (rm3: fb_docs={self.fb_docs}, "
            f"fb_terms={self.fb_terms}, orig_w={self.orig_weight})"
        )


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_madr(qrels: Dict[str, Dict[str, int]],
                 ranklists: Dict[str, Dict[str, float]],
                 cutoffs: List[int]) -> Dict[str, float]:
    madr_results = {f"MADR_{k}": 0.0 for k in cutoffs}
    for qid, rel_docs in qrels.items():
        if qid not in ranklists:
            continue
        retrieved = ranklists[qid]
        ranked = sorted(retrieved.keys(), key=lambda d: retrieved[d], reverse=True)
        for cutoff in cutoffs:
            top_k = ranked[:cutoff]
            ranks = [r for r,(docid) in enumerate(top_k, start=1) if docid in rel_docs and rel_docs[docid] > 0]
            adr = sum(ranks)/len(ranks) if ranks else cutoff + 1
            madr_results[f"MADR_{cutoff}"] += adr
    # Average over the evaluated queries (i.e., those in ranklists)
    num_queries = len(ranklists)
    for key in madr_results:
        madr_results[key] /= num_queries
    return madr_results


def evaluate(ranklists: Dict[str, Dict[str, float]],
             qrels: Dict[str, Dict[str, int]]) -> Metrics:
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {
            'P.3', 'P.5', 'P.10',
            'recall.3', 'recall.5', 'recall.10',
            'map_cut.3', 'map_cut.5', 'map_cut.10'
        }
    )
    results = evaluator.evaluate(ranklists)
    P_3=P_5=P_10=R_3=R_5=R_10=MAP_3=MAP_5=MAP_10=0.0
    for qid, scores in results.items():
        P_3 += scores['P_3']; P_5 += scores['P_5']; P_10 += scores['P_10']
        R_3 += scores['recall_3']; R_5 += scores['recall_5']; R_10 += scores['recall_10']
        MAP_3 += scores['map_cut_3']; MAP_5 += scores['map_cut_5']; MAP_10 += scores['map_cut_10']
    n = len(ranklists)
    madr = compute_madr(qrels, ranklists, [3,5,10])
    return Metrics(P_3/n,P_5/n,P_10/n,R_3/n,R_5/n,R_10/n,MAP_3/n,MAP_5/n,MAP_10/n,
                   madr['MADR_3'], madr['MADR_5'], madr['MADR_10'])


def build_ranklists(searcher: LuceneSearcher,
                    claims: List[Dict[str, str]],
                    top_k: int) -> Dict[str, Dict[str, float]]:
    queries = [c['input'] for c in claims]
    qids = [c['id'] for c in claims]
    hits = searcher.batch_search(queries, qids=qids, k=top_k, threads=8)
    ranklists: Dict[str, Dict[str, float]] = {}
    for cid, curr_hits in hits.items():
        retrieved: Dict[str, float] = {}
        for h in curr_hits:
            retrieved[h.docid] = float(h.score)
        ranklists[cid] = retrieved
    return ranklists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-size', type=int, default=3000,
                    help='Number of queries (claims) to sample for optimization.')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--rm3', action='store_true', help='Evaluate RM3 feedback as well.')
    ap.add_argument('--fb-docs', type=int, default=10)
    ap.add_argument('--fb-terms', type=int, default=10)
    ap.add_argument('--orig-weight', type=float, default=0.7)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--objective', type=str, default='composite',
                    choices=['composite','map10','p3','p5','p10','madr3','madr5','madr10'],
                    help='Optimization objective (default: composite).')
    args = ap.parse_args()

    if not (os.path.exists(QRELS_PATH) and os.path.exists(CLAIMS_PATH)):
        raise SystemExit('QRELS or CLAIMS missing. Run tests/test_bm25 first to generate them.')

    qrels = load_json(QRELS_PATH)
    claims_full = load_json(CLAIMS_PATH)
    claims = claims_full[:args.sample_size]

    param_grid: List[ParamSet] = []
    k1_vals = [0.9, 1.0, 1.2, 1.4, 1.6]
    b_vals = [0.40, 0.55, 0.75, 0.90]
    for k1 in k1_vals:
        for b in b_vals:
            param_grid.append(ParamSet(k1=k1, b=b, rm3=False))
            if args.rm3:
                param_grid.append(ParamSet(k1=k1, b=b, rm3=True,
                                            fb_docs=args.fb_docs,
                                            fb_terms=args.fb_terms,
                                            orig_weight=args.orig_weight))

    results: List[Tuple[ParamSet, Metrics]] = []

    print(f"Optimizing over {len(param_grid)} parameter sets using {len(claims)} queries...")
    searcher = LuceneSearcher(str(INDEX_DIR))

    for ps in param_grid:
        # Reset base BM25 params
        searcher.set_bm25(ps.k1, ps.b)
        if ps.rm3:
            searcher.set_rm3(ps.fb_terms, ps.fb_docs, ps.orig_weight)
        else:
            if searcher.is_using_rm3():
                searcher.unset_rm3()

        ranklists = build_ranklists(searcher, claims, args.top_k)
        metrics = evaluate(ranklists, qrels)
        results.append((ps, metrics))
        if args.verbose:
            print(f"{ps.label()} -> MAP@10={metrics.MAP_10:.3f}, MADR@10={metrics.MADR_10:.3f}, P@3={metrics.P_3:.3f}, P@10={metrics.P_10:.3f}, score={metrics.score(args.objective):.4f}")

    # Sort by composite score descending
    results.sort(key=lambda x: x[1].score(args.objective), reverse=True)

    print('\n=== Top 10 Parameter Sets (by composite score) ===')
    for i, (ps, m) in enumerate(results[:10], start=1):
        print(f"{i:2}. {ps.label()} | MAP@10={m.MAP_10:.3f} MADR@10={m.MADR_10:.3f} P@3={m.P_3:.3f} P@10={m.P_10:.3f} R@10={m.R_10:.3f} score={m.score(args.objective):.4f}")

    best_ps, best_m = results[0]
    print('\n=== Recommended Parameters ===')
    print(best_ps.label())
    print(f"MAP@10={best_m.MAP_10:.3f}, MADR@10={best_m.MADR_10:.3f}, Precision@10={best_m.P_10:.3f}, Recall@10={best_m.R_10:.3f}")
    print('\nTo apply: update searcher.set_bm25({:.2f}, {:.2f}){} in ragar_corag.py'.format(
        best_ps.k1, best_ps.b,
        (" and searcher.set_rm3({},{},{})".format(best_ps.fb_terms, best_ps.fb_docs, best_ps.orig_weight) if best_ps.rm3 else "")
    ))

if __name__ == '__main__':
    main()