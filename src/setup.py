import os
import sys
from pathlib import Path
from config import DATA_DIR, CLAIMS_PATH, QRELS_PATH, RANKLISTS_PATH, INDEX_DIR

# Add project root to path to find tests
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tests.utils import write_qrels, write_ranklists
from pyserini.search.lucene import LuceneSearcher

def main():
    print("Using tests.utils to generate standard training data")
    
    # 1. Generate QRELS and CLAIMS (this uses the filtering logic in utils.py)
    # We limit to 100 for our quick comparison, but this uses the function.
    # If we wanted the full dataset, we would just remove max_claims.
    print(f"Generating claims/qrels to {DATA_DIR}...")
    write_qrels(
        qrels_path=str(QRELS_PATH),
        claims_path=str(CLAIMS_PATH),
        data_dir=str(DATA_DIR),
        max_claims=50  # Configurable for testing
    )
    
    # 2. Generate Ranklists (BM25 retrieval)
    print("Loading Lucene Searcher...")
    try:
        searcher = LuceneSearcher(str(INDEX_DIR))
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Make sure you have run index.py first!")
        sys.exit(1)

    import json
    with open(CLAIMS_PATH, 'r', encoding='utf-8') as f:
        claims_list = json.load(f)

    print(f"Generating ranklists for {len(claims_list)} claims...")
    write_ranklists(
        lucene_searcher=searcher,
        num_worker=1,
        ranklist_path=str(RANKLISTS_PATH),
        raw_claims=claims_list,
        top_k=100 
    )
    
    print("Data setup complete!")

if __name__ == "__main__":
    main()