"""
Compare baseline vs tuned retrieval metrics in terminal tables.

Baseline uses precomputed ranklists (tests/test_bm25 output).
Tuned is computed on-the-fly with BM25 (and optional RM3) for the same sampled queries.

Usage:
  python scripts/compare_metrics.py --k1 0.9 --b 0.4 --sample-size 5000 [--rm3 --fb-docs 10 --fb-terms 10 --orig-weight 0.8]

Tables printed:
Precision (P@3,5,10)
Recall (R@3,5,10)
MAP (MAP@3,5,10)
MADR (MADR@3,5,10)  [lower is better]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root on path
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import QRELS_PATH, CLAIMS_PATH, RANKLISTS_PATH, INDEX_DIR
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval


def load_json(path: Path) -> dict:
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
            ranks = [rank for rank, docid in enumerate(top_k, start=1)
                     if docid in rel_docs and rel_docs[docid] > 0]
            adr = sum(ranks)/len(ranks) if ranks else cutoff + 1
            madr_results[f"MADR_{cutoff}"] += adr
    n = len(ranklists)
    for key in madr_results:
        madr_results[key] = madr_results[key] / n if n else 0.0
    return madr_results


def evaluate(ranklists: Dict[str, Dict[str, float]],
             qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {'P.3','P.5','P.10','recall.3','recall.5','recall.10','map_cut.3','map_cut.5','map_cut.10'}
    )
    results = evaluator.evaluate(ranklists)
    P_3=P_5=P_10=R_3=R_5=R_10=MAP_3=MAP_5=MAP_10=0.0
    for qid, scores in results.items():
        P_3 += scores['P_3']; P_5 += scores['P_5']; P_10 += scores['P_10']
        R_3 += scores['recall_3']; R_5 += scores['recall_5']; R_10 += scores['recall_10']
        MAP_3 += scores['map_cut_3']; MAP_5 += scores['map_cut_5']; MAP_10 += scores['map_cut_10']
    n = len(ranklists) or 1
    metrics = {
        'P_3': P_3/n, 'P_5': P_5/n, 'P_10': P_10/n,
        'R_3': R_3/n, 'R_5': R_5/n, 'R_10': R_10/n,
        'MAP_3': MAP_3/n, 'MAP_5': MAP_5/n, 'MAP_10': MAP_10/n,
    }
    metrics.update(compute_madr(qrels, ranklists, [3,5,10]))
    return metrics


def build_ranklists(searcher: LuceneSearcher,
                    claims: List[Dict[str, str]],
                    top_k: int = 10) -> Dict[str, Dict[str, float]]:
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


def filter_to_qids(d: Dict[str, dict], qids: List[str]) -> Dict[str, dict]:
    qset = set(qids)
    return {k: v for k, v in d.items() if k in qset}


def print_table(title: str,
                rows: List[Tuple[str, float, float, float, float]],
                lower_is_better: bool = False) -> None:
    print(f"\n{title}")
    print("-" * 66)
    print(f"{'Metric':<10} {'Baseline':>10} {'Tuned':>10} {'Delta':>10} {'Change':>10}")
    print("-" * 66)
    for name, base, tuned, delta, change in rows:
        arrow = '↓' if lower_is_better and delta < 0 else ('↑' if (not lower_is_better and delta > 0) else ' ')
        print(f"{name:<10} {base:>10.3f} {tuned:>10.3f} {delta:>10.3f} {change:>9.1f}%{arrow}")


def main():
    ap = argparse.ArgumentParser(description='Compare baseline vs tuned retrieval metrics with tables.')
    ap.add_argument('--k1', type=float, required=True, help='BM25 k1 for tuned')
    ap.add_argument('--b', type=float, required=True, help='BM25 b for tuned')
    ap.add_argument('--rm3', action='store_true', help='Enable RM3 for tuned')
    ap.add_argument('--fb-docs', type=int, default=10)
    ap.add_argument('--fb-terms', type=int, default=10)
    ap.add_argument('--orig-weight', type=float, default=0.7)
    ap.add_argument('--sample-size', type=int, default=5000, help='Use first N claims for fair comparison')
    ap.add_argument('--top-k', type=int, default=10)
    args = ap.parse_args()

    # Load data
    qrels = load_json(QRELS_PATH)
    claims_full = load_json(CLAIMS_PATH)
    claims = claims_full[:args.sample_size]
    qids = [c['id'] for c in claims]

    # Baseline: precomputed ranklists filtered to qids
    base_ranklists_full = load_json(RANKLISTS_PATH)
    base_ranklists = filter_to_qids(base_ranklists_full, qids)
    base_qrels = filter_to_qids(qrels, qids)
    base_metrics = evaluate(base_ranklists, base_qrels)

    # Tuned: compute search with given k1,b (and optional RM3) on same qids
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(args.k1, args.b)
    if args.rm3:
        searcher.set_rm3(args.fb_terms, args.fb_docs, args.orig_weight)
    tuned_ranklists = build_ranklists(searcher, claims, top_k=args.top_k)
    tuned_metrics = evaluate(tuned_ranklists, base_qrels)

    # Build tables
    def rows_for(keys: List[str], lower_is_better: bool=False):
        rows = []
        for k in keys:
            base = base_metrics[k]
            tuned = tuned_metrics[k]
            delta = tuned - base
            change = (delta / base * 100.0) if base != 0 else 0.0
            rows.append((k.replace('_', '@'), base, tuned, delta, change))
        return rows

    prec_rows = rows_for(['P_3','P_5','P_10'])
    rec_rows  = rows_for(['R_3','R_5','R_10'])
    map_rows  = rows_for(['MAP_3','MAP_5','MAP_10'])
    madr_rows = rows_for(['MADR_3','MADR_5','MADR_10'], lower_is_better=True)

    print(f"\nComparing on {len(qids)} queries (same subset for baseline and tuned)")
    print(f"Tuned params: k1={args.k1}, b={args.b}{' with RM3' if args.rm3 else ''}\n")
    print_table('Precision', prec_rows)
    print_table('Recall', rec_rows)
    print_table('MAP', map_rows)
    print_table('MADR (lower is better)', madr_rows, lower_is_better=True)

if __name__ == '__main__':
    main()