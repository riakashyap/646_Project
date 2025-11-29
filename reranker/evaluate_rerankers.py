import os
import sys
import json
import torch
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- 1. Set JAVA_HOME explicitly for Pyserini ---
# Check environment variable first
if "JAVA_HOME" not in os.environ:
    # Try standard Linux/Windows paths
    candidates = [
        "/usr/lib/jvm/java-21-openjdk-amd64",  # Linux default
        "/usr/lib/jvm/java-17-openjdk-amd64",
        r"C:\Program Files\Java\jdk-21",        # Windows default
        r"C:\Program Files\Microsoft\jdk-21.0.9.10-hotspot" # Your machine specific
    ]
    for c in candidates:
        if os.path.exists(c):
            os.environ["JAVA_HOME"] = c
            print(f"[Setup] Automatically set JAVA_HOME to {c}")
            break
    
    if "JAVA_HOME" not in os.environ:
        print("[Warning] JAVA_HOME not found. Pyserini might fail.")

# Add src to path to find config
sys.path.append(str(Path(__file__).parent))

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    print("Error: Pyserini not found. Please install it.")
    sys.exit(1)
except Exception as e:
    print(f"Error importing Pyserini (JAVA_HOME issue?): {e}")
    print(f"Current JAVA_HOME: {os.environ.get('JAVA_HOME')}")
    sys.exit(1)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from reranker.e2rank_reranker import E2RankReranker
from src.config import INDEX_DIR, CLAIMS_PATH, QRELS_PATH

def calculate_metrics(run, qrels, k_values=[1, 5, 10]):
    """Calculate Precision@K, Recall@K, MAP, NDCG@K"""
    metrics = {f"P@{k}": 0.0 for k in k_values}
    metrics.update({f"R@{k}": 0.0 for k in k_values})
    metrics["MAP"] = 0.0
    metrics["NDCG"] = 0.0
    
    cnt = 0
    for qid, ranked_list in run.items():
        if qid not in qrels:
            continue
            
        cnt += 1
        relevant_docs = set(qrels[qid])
        
        # Precision & Recall
        for k in k_values:
            retrieved = [doc_id for doc_id, _ in ranked_list[:k]]
            num_relevant = sum(1 for doc in retrieved if doc in relevant_docs)
            metrics[f"P@{k}"] += num_relevant / k
            metrics[f"R@{k}"] += num_relevant / len(relevant_docs) if relevant_docs else 0
            
        # MAP
        ap = 0.0
        hits = 0
        for i, (doc_id, _) in enumerate(ranked_list):
            if doc_id in relevant_docs:
                hits += 1
                ap += hits / (i + 1)
        metrics["MAP"] += ap / len(relevant_docs) if relevant_docs else 0
        
        # NDCG
        dcg = 0.0
        idcg = sum(1 / np.log2(i + 2) for i in range(len(relevant_docs)))
        for i, (doc_id, _) in enumerate(ranked_list):
            if doc_id in relevant_docs:
                dcg += 1 / np.log2(i + 2)
        metrics["NDCG"] += dcg / idcg if idcg > 0 else 0

    for k in metrics:
        metrics[k] /= max(1, cnt)
        
    return metrics

def evaluate_model(model_type, model_path, claims, qrels, searcher, limit=None):
    print(f"\n--- Evaluating {model_type} ---")
    
    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if model_type == "Pairwise":
        tokenizer = AutoTokenizer.from_pretrained("naver/trecdl22-crossencoder-debertav3")
        model = AutoModelForSequenceClassification.from_pretrained("naver/trecdl22-crossencoder-debertav3").to(device)
        model.eval()
    else:
        # E2Rank
        reranker = E2RankReranker(model_path="naver/trecdl22-crossencoder-debertav3", device=device, use_layerwise=True)
    
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
            inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().tolist()
                if not isinstance(scores, list): scores = [scores]
            
            scored_docs = sorted(zip([d[0] for d in docs], scores), key=lambda x: x[1], reverse=True)
            
        else: # E2Rank
            # E2RankReranker.rerank expects [(id, text)]
            results = reranker.rerank(query, docs, top_k=len(docs))
            # results is [(id, text, score)]
            scored_docs = [(r[0], r[2]) for r in results]
            
        run_results[qid] = scored_docs
        
        if (i+1) % 5 == 0:
             print(f"Processed {i+1}/{len(claim_items)} claims... ({(time.time() - start_time)/(i+1):.2f}s/claim)")

    return calculate_metrics(run_results, qrels)

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
    m1 = evaluate_model("Pairwise", "output/pairwise_small", claims, qrels, searcher)
    
    # 2. E2Rank Layerwise
    m2 = evaluate_model("E2Rank", "output/e2rank_small", claims, qrels, searcher)

    print("\n\n=== FINAL COMPARISON ===")
    print(f"{'Metric':<10} | {'Pairwise':<10} | {'E2Rank':<10}")
    print("-" * 36)
    for k in ["MAP", "NDCG", "P@1", "P@5"]:
        print(f"{k:<10} | {m1.get(k,0):.4f}     | {m2.get(k,0):.4f}")

if __name__ == "__main__":
    main()
