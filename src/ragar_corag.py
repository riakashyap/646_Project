from .corag import Corag
from .model_clients import ModelClient
from .retrieval import INDEX_DIR
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
        return self._mc.send_prompt("init_question", [claim])

    def answer(self, question: str) -> str:
        hits = self._searcher.search(question, k=3)
        search_results = []
        for hit in hits:
            doc = self._searcher.doc(hit.docid)
            contents = doc.get("contents")
            search_results.append(contents)

        output = "\n\n".join(search_results)
        return self._mc.send_prompt("answer", [output, question])

    def next_question(self, claim: str, qa_pairs: list[tuple[str, str]]) -> str:
        return self._mc.send_prompt("next_question", [claim, qa_pairs])

    def stop_check(self, claim: str, qa_pairs: list[tuple[str, str]]) -> bool:
        res = self._mc.send_prompt("stop_check", [claim, qa_pairs]).lower()

        # Stops if then model was indecisive or gave a non-binary answer
        return "true" in res 

    def verdict(self, claim: str, qa_pairs: list[tuple[str, str]]) -> tuple[int, str | None]:
        res = self._mc.send_prompt("verdict", [claim, qa_pairs])
        verdict = None

        # TODO: Ideally need to standardize these (and the stop_check) across the prompts.
        # TODO: define an enum or Verdict class for this
        # TODO: could also use a map, and extract this to a parsers.py file for easier reuse
        # 0 -> refutes, 1 -> supports, 2 -> not enough evidence
        lower = res.lower()
        if "refute" in lower:
            verdict = 0
        elif "support" in lower:
            verdict = 1
        elif "fail" in lower:
            verdict = 2
        
        return verdict, res
