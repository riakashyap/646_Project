"""
Copyright:

  Copyright © 2025 Ananya-Jha-code

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/
"""
from tests.utils import eval_on_fever
from src.config import INDEX_DIR, CLAIMS_PATH, QRELS_PATH
from reranker.e2rank_reranker import E2RankReranker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import sys
import json
import torch
import time
from tqdm import tqdm
from pathlib import Path

# --- 1. Set JAVA_HOME explicitly for Pyserini ---
if "JAVA_HOME" not in os.environ:
    # PR Feedback: Don't guess paths. Instruct the user to set it.
    print("\n[ERROR] JAVA_HOME environment variable is not set.")
    print("Pyserini requires a Java 11+ (ideally 21) JDK to be installed and JAVA_HOME set.")
    print("\nTo set it:")
    print("  - Windows (PowerShell): $env:JAVA_HOME = 'C:\\Path\\To\\JDK'")
    print("  - Windows (CMD): set JAVA_HOME=C:\\Path\\To\\JDK")
    print("  - Linux/macOS: export JAVA_HOME=/path/to/jdk")
    print("\nPlease set JAVA_HOME and try again.\n")
    sys.exit(1)

# Add project root to path to find config and tests
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    print("Error: Pyserini not found. Please install it.")
    sys.exit(1)
except Exception as e:
    print(f"Error importing Pyserini (JAVA_HOME issue?): {e}")
    print(f"Current JAVA_HOME: {os.environ.get('JAVA_HOME')}")
    sys.exit(1)


def evaluate_model(model_type, model_path, claims, qrels, searcher, limit=None):
    print(f"\n--- Evaluating {model_type} ---")

    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if model_type == "Pairwise":
        tokenizer = AutoTokenizer.from_pretrained(
            "naver/trecdl22-crossencoder-debertav3")
        model = AutoModelForSequenceClassification.from_pretrained(
            "naver/trecdl22-crossencoder-debertav3").to(device)
        model.eval()
    else:
        # E2Rank
        reranker = E2RankReranker(
            model_path="naver/trecdl22-crossencoder-debertav3", device=device, use_layerwise=True)

    run_results = {}

    # Process claims
    claim_items = list(claims.items())
    if limit:
        claim_items = claim_items[:limit]

    start_time = time.time()

    for i, (qid, claim_data) in enumerate(claim_items):
        query = claim_data['claim']

        # 1. Retrieve with BM25 (Limit to top 20 for speed on CPU)
        hits = searcher.search(query, k=20)
        docs = []
        for hit in hits:
            try:
                # Fetch full document object using docid
                doc_obj = searcher.doc(hit.docid)
                if doc_obj is None:
                    continue

                try:
                    # Try parsing as JSON first
                    json_content = json.loads(doc_obj.raw())
                    content = json_content.get('contents', doc_obj.contents())
                except:
                    # Fallback to plain contents
                    content = doc_obj.contents()
            except Exception:
                content = ""

            docs.append((hit.docid, content))

        if not docs:
            continue

        # 2. Rerank
        scored_docs = []
        if model_type == "Pairwise":
            pairs = [[query, d[1]] for d in docs]
            inputs = tokenizer(pairs, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().tolist()
                if not isinstance(scores, list):
                    scores = [scores]

            scored_docs = sorted(
                zip([d[0] for d in docs], scores), key=lambda x: x[1], reverse=True)

        else:  # E2Rank
            # E2RankReranker.rerank expects [(id, text)]
            results = reranker.rerank(query, docs, top_k=len(docs))
            # results is [(id, text, score)]
            scored_docs = [(r[0], r[2]) for r in results]

        run_results[qid] = scored_docs

        if (i+1) % 5 == 0:
            print(
                f"Processed {i+1}/{len(claim_items)} claims... ({(time.time() - start_time)/(i+1):.2f}s/claim)")

    # Convert run_results to dictionary format for eval_on_fever
    # run_results: {qid: [(docid, score), ...]}
    # ranklists: {qid: {docid: score}}
    ranklists = {}
    for qid, docs in run_results.items():
        ranklists[qid] = {docid: score for docid, score in docs}

    return eval_on_fever(qrels, ranklists)


def main():
    print("Loading Resources...")
    try:
        searcher = LuceneSearcher(str(INDEX_DIR))
    except Exception as e:
        print(f"Failed to load index from {INDEX_DIR}: {e}")
        return

    with open(CLAIMS_PATH, 'r', encoding='utf-8') as f:
        claims = json.load(f)

    with open(QRELS_PATH, 'r', encoding='utf-8') as f:
        qrels = json.load(f)

    print(f"Loaded {len(claims)} claims and {len(qrels)} qrels.")

    # Process all claims in the file (the file is already limited by prepare_training_data.py)
    # But we pass a small limit if needed for debugging
    # limit = 5

    # 1. Pairwise Baseline
    m1 = evaluate_model("Pairwise", "output/pairwise_small",
                        claims, qrels, searcher)

    # 2. E2Rank Layerwise
    m2 = evaluate_model("E2Rank", "output/e2rank_small",
                        claims, qrels, searcher)

    print("\n\n=== FINAL COMPARISON ===")
    print(f"{'Metric':<10} | {'Pairwise':<10} | {'E2Rank':<10}")
    print("-" * 36)
    # eval_on_fever returns keys like "P_5", "R_5", "MAP_5"
    metrics_to_show = ["MAP_10", "P_5", "R_5"]
    for k in metrics_to_show:
        print(f"{k:<10} | {m1.get(k, 0):.4f}     | {m2.get(k, 0):.4f}")


if __name__ == "__main__":
    main()
