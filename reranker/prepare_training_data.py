"""
Copyright:

  Copyright © 2025 Ananya-Jha-code

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/
""" 
import json
import argparse
import random
from pathlib import Path
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import sys

# Add src to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from src.config import CLAIMS_PATH, QRELS_PATH, RANKLISTS_PATH, INDEX_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Limit number of claims for fast testing")
    args = parser.parse_args()

    print(f"Loading FEVER dataset (limit={args.limit})...")
    try:
        dataset = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Initializing Lucene Searcher from {INDEX_DIR}...")
    try:
        searcher = LuceneSearcher(str(INDEX_DIR))
    except Exception as e:
        print(f"Error initializing searcher: {e}")
        return

    claims = {}
    qrels = {}
    ranklists = {}

    count = 0
    for item in tqdm(dataset):
        if item['label'] not in ['SUPPORTS','REFUTES']: 
            continue
            
        claim_id = str(item['id'])
        claim_text = item['claim']
        
        # Extract evidence (ground truth)
        # FEVER evidence format: [ [ [annotation_id, evidence_id, wiki_url, sentence_index], ... ], ... ]
        # We just need the wiki_url (title) which maps to our docid
        evidence_sets = item['evidence']
        relevant_docs = set()
        for evidence_set in evidence_sets:
            for evidence in evidence_set:
                # evidence[2] is the wiki title/url
                # Pyserini index usually stores titles as they are or normalized.
                # We need to check how index.py stored them.
                # Assuming index stored them as is.
                if evidence[2]:
                    relevant_docs.add(evidence[2])
        
        if not relevant_docs:
            continue

        # Add to our datasets
        claims[claim_id] = {"claim": claim_text}
        qrels[claim_id] = list(relevant_docs)
        
        # Generate Ranklist (BM25 Negatives)
        hits = searcher.search(claim_text, k=20)
        ranklists[claim_id] = [hit.docid for hit in hits]
        
        count += 1
        if count >= args.limit:
            break

    print(f"Saving {len(claims)} claims to {CLAIMS_PATH}")
    with open(CLAIMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(claims, f)

    print(f"Saving qrels to {QRELS_PATH}")
    with open(QRELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(qrels, f)

    print(f"Saving ranklists to {RANKLISTS_PATH}")
    with open(RANKLISTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(ranklists, f)

    print("Done generating training data.")

if __name__ == "__main__":
    main()

