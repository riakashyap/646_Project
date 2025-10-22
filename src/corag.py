"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file includes a class abstracting a CoRAG pipeline.

Code:
"""


from abc import ABC, abstractmethod

class Corag(ABC):

    @abstractmethod
    def init_question(self, claim: str) -> str:
        pass

    @abstractmethod
    def answer(self, question: str) -> str:
        pass

    @abstractmethod
    def next_question(self, claim: str, qa_pairs: list[tuple[str, str]]) -> str:
        pass

    @abstractmethod
    def stop_check(self, claim: str, qa_pairs: list[tuple[str, str]]) -> bool:
        pass

    @abstractmethod
    def verdict(self, claim: str, qa_pairs: list[tuple[str, str]]) -> tuple[int, str]:
        pass

    def run(self, claim: str, max_iters: int = 3) -> dict[str, any]:
        qa_pairs = []
        question = self.init_question(claim)

        for i in range(max_iters):
            if i > 0:
                question = self.next_question(claim, qa_pairs)
            answer = self.answer(question)
            qa_pairs.append((question, answer))
            if self.stop_check(claim, qa_pairs):
                break

        verdict, raw = self.verdict(claim, qa_pairs)
        return {
            "claim": claim,
            "qa_pairs": qa_pairs,
            "verdict": verdict,
            "verdict_raw": raw
        }
