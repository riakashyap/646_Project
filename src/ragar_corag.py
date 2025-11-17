"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file includes a class implementing a RAG CoRAG pipeline.

Code:
"""

from .corag import Corag
from .model_clients import ModelClient
from .config import INDEX_DIR
from pyserini.search.lucene import LuceneSearcher

class RagarCorag(Corag):
    _mc: ModelClient
    _searcher: LuceneSearcher

    def __init__(self, mc: ModelClient):
        super().__init__()
        self._mc = mc
        self._searcher = LuceneSearcher(str(INDEX_DIR)) # Ideally would inject this but I'm lazy
        self._searcher.set_bm25(1.2, 0.75)

    def init_question(self, claim: str) -> str:
        return self._mc.send_prompt("init_question", [claim]).strip()

    def answer(self, question: str) -> str:
        hits = self._searcher.search(question, k=3)
        search_results = []
        for hit in hits:
            doc = self._searcher.doc(hit.docid)
            contents = doc.get("contents")
            search_results.append(contents)

        output = "\n\n".join(search_results)
        return self._mc.send_prompt("answer", [output, question]).strip()

    def next_question(self, claim: str, qa_pairs: list[tuple[str, str]]) -> str:
        return self._mc.send_prompt("next_question", [claim, qa_pairs]).strip()

    def stop_check(self, claim: str, qa_pairs: list[tuple[str, str]]) -> bool:
        res = self._mc.send_prompt("stop_check", [claim, qa_pairs]).lower()

        has_inconclusive = "inconclusive" in res
        has_conclusive = "conclusive" in res and not has_inconclusive

        if has_conclusive and not has_inconclusive:
            return True
        return False

    def verdict(self, claim: str, qa_pairs: list[tuple[str, str]]) -> tuple[int, str | None]:
        res = self._mc.send_prompt("verdict", [claim, qa_pairs])
        verdict = None

        # TODO: define an enum or Verdict class for this
        # TODO: could also use a map, and extract this to a parsers.py file for easier reuse
        # 0 -> false, 1 -> true, 2 -> inconclusive
        lower = res.lower()
        if "false" in lower:
            verdict = 0
        elif "true" in lower:
            verdict = 1
        elif "inconclusive" in lower:
            verdict = 2

        return verdict, res
