# src/agentic_rag/retriever.py
from pyserini.search.lucene import LuceneSearcher
from src.index import INDEX_DIR

class BM25Retriever:
    def __init__(self, index_dir=INDEX_DIR, k1=1.2, b=0.75):
        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_bm25(k1=k1, b=b)

    def retrieve(self, claim, top_k=3):
        hits = self.searcher.search(claim, k=top_k)
        docs = [self.searcher.doc(h.docid).get("contents") for h in hits]
        return docs
