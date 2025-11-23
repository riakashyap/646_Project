"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric
  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file includes a class implementing a RAG CoRAG pipeline.

Code:
"""

from .corag import Corag
from .model_clients import ModelClient
from . import config
from pyserini.search.lucene import LuceneSearcher
from .parsers import parse_ternary, parse_conclusive
from .madr import run_madr

class RagarCorag(Corag):
    _mc: ModelClient
    _searcher: LuceneSearcher
    _debate_stop: bool

    def __init__(self, mc: ModelClient, debate_stop: bool, reranker=None):
        super().__init__()
        self._mc = mc
        self._debate_stop = debate_stop
        self._searcher = LuceneSearcher(str(config.INDEX_DIR))
        self._searcher.set_bm25(1.2, 0.75)
        self._reranker = reranker
        
        if self._reranker is not None:
            print(f"Reranker enabled: {self._reranker}")

    def init_question(self, claim: str) -> str:
        return self._mc.send_prompt("init_question", [claim]).strip()

    def answer(self, question: str) -> str:
        bm25_k = 50 if self._reranker is not None else 3 
        ## TODO: Finalise after a few iterative tests
        
        hits = self._searcher.search(question, k=bm25_k)
        search_results = []
        
        if self._reranker is not None:
            docs = []
            for hit in hits:
                doc = self._searcher.doc(hit.docid)
                contents = doc.get("contents")
                if contents:
                    docs.append((hit.docid, contents))
            
            if docs:
                reranked = self._reranker.rerank(
                    question, docs
                )
                search_results = [contents for _, contents, _ in reranked]
        else:
            # Use BM25 results directly
            for hit in hits:
                doc = self._searcher.doc(hit.docid)
                contents = doc.get("contents")
                if contents:
                    search_results.append(contents)
        
        output = "\n\n".join(search_results)
        return self._mc.send_prompt("answer", [output, question]).strip()


    def next_question(self, claim: str, qa_pairs: list[tuple[str, str]]) -> str:
        return self._mc.send_prompt("next_question", [claim, qa_pairs]).strip()

    def stop_check(self, claim: str, qa_pairs: list[tuple[str, str]]) -> bool:
        exp = self._mc.send_prompt("stop_check", [claim, qa_pairs])
        exp_bool = parse_conclusive(exp)

        if not self._debate_stop:
            return exp_bool

        exp_bool_refined = parse_conclusive(
            run_madr(self._mc, claim, qa_pairs, exp)
        )

        if exp_bool != exp_bool_refined:
            config.LOGGER and config.LOGGER.info(f"MADR swapped to {exp_bool_refined}")

        return exp_bool_refined

    def verdict(self, claim: str, qa_pairs: list[tuple[str, str]]) -> tuple[int, str | None]:
        res = self._mc.send_prompt("verdict", [claim, qa_pairs])
        return parse_ternary(res), res
