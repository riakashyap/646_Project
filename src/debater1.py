import re
from typing import List, Dict, Optional
from .error_typology import ErrorTypology

class Debater1:
    SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    EVENT_RE = re.compile(r"\b(?:was|were|is|are|has|have|did|did not|performed|conducted|created|authored|posted|published)\b", re.IGNORECASE)

    def __init__(self, explanation: str, evidence: str):
        self.explanation = (explanation or "").strip()
        self.evidence = (evidence or "").strip()
        self.sentences = self._split_sentences(self.explanation)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        return re.split(Debater1.SENT_SPLIT_RE, text) if text else []

    @staticmethod
    def _extract_proper_nouns(text: str) -> List[str]:
        """Extract capitalized words and remove common words. This is useful for identifying Intrinsic and Extrinsic Entity Errors"""
        tokens = re.findall(r"\b[A-Z][a-zA-Z0-9'\-]+(?:\s+[A-Z][a-zA-Z0-9'\-]+)*\b", text)
        common_starts = {"The", "It", "A", "An", "In", "On", "By", "As", "To"}
        return [t for t in tokens if t not in common_starts]

    @staticmethod
    def _extract_noun_phrases(text: str) -> List[str]:
        """Extracts multi-word capitalized noun phrases from the text."""
        return re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)

    @staticmethod
    def _content_words(text: str) -> List[str]:
        """Extracts lower-cased content words (length ≥ 4) from the text. Used to detect Irrelevant Evidence Errors by comparing the words in a
        sentence with the words found in the evidence"""
        return re.findall(r"\b[a-z]{4,}\b", text.lower())

    @staticmethod
    def _extract_events(text: str) -> List[str]:
        """Extracts event-related phrases from the text. Helps to detect Intrinsic and Extrinsic Event Errors."""
        return re.findall(Debater1.EVENT_RE, text)

    def analyze(self, prev_feedback: Optional[List[Dict]] = None) -> List[Dict]:
        feedback: List[Dict] = []

        expl_entities = set(self._extract_proper_nouns(self.explanation))
        ev_entities = set(self._extract_proper_nouns(self.evidence))

        expl_phrases = set(self._extract_noun_phrases(self.explanation))
        ev_phrases = set(self._extract_noun_phrases(self.evidence))

        expl_events = set(self._extract_events(self.explanation))
        ev_events = set(self._extract_events(self.evidence))

        def add_item(error_type, sent_idx, excerpt, description, suggestion, corrected=None):
            item = {
                "error_type": error_type,
                "sentence_index": sent_idx,
                "sentence": self.sentences[sent_idx] if 0 <= sent_idx < len(self.sentences) else "",
                "excerpt": excerpt,
                "description": description,
                "suggestion": suggestion,
            }
            if corrected:
                item["corrected"] = corrected
            feedback.append(item)

        # Extrinsic Entity Error (introduces an entity that does not exist in the evidence.)
        for ent in sorted(expl_entities - ev_entities):
            for i, s in enumerate(self.sentences):
                if ent in s:
                    add_item(
                        ErrorTypology.EXTRINSIC_ENTITY_ERROR,
                        i,
                        ent,
                        f"Entity '{ent}' appears in the explanation but is not in the evidence.",
                        f"Verify whether '{ent}' is supported by the evidence; remove or qualify if not."
                    )
                    break

        # Intrinsic Entity Error (entity is mentioned in both the explanation and evidence, but wrong details)
        if expl_entities & ev_entities and expl_entities != ev_entities:
            for ent in sorted(expl_entities):
                if ent not in ev_entities:
                    continue
                for i, s in enumerate(self.sentences):
                    if ent in s:
                        corrected = s.replace(ent, next(iter(ev_entities)))
                        add_item(
                            ErrorTypology.INTRINSIC_ENTITY_ERROR,
                            i,
                            ent,
                            "Named entities differ from evidence; possible misattribution.",
                            "Confirm correct named entities from evidence.",
                            corrected=corrected
                        )
                        break

        # Extrinsic noun phrases
        for ph in sorted(expl_phrases - ev_phrases):
            for i, s in enumerate(self.sentences):
                if ph in s:
                    add_item(
                        ErrorTypology.EXTRINSIC_NOUN_PHRASE_ERROR,
                        i,
                        ph,
                        f"Noun phrase '{ph}' appears in explanation but not in evidence.",
                        "Either provide evidence or remove/qualify."
                    )
                    break

        # Intrinsic noun phrases
        for ph in sorted(expl_phrases & ev_phrases):
            for i, s in enumerate(self.sentences):
                if ph in s and ph not in ev_phrases:
                    add_item(
                        ErrorTypology.INTRINSIC_NOUN_PHRASE_ERROR,
                        i,
                        ph,
                        f"Noun phrase '{ph}' misrepresented compared to evidence.",
                        "Verify the correct noun phrase from evidence."
                    )
                    break

        # Extrinsic events
        for ev in sorted(expl_events - ev_events):
            for i, s in enumerate(self.sentences):
                if ev in s:
                    add_item(
                        ErrorTypology.EXTRINSIC_EVENT_ERROR,
                        i,
                        ev,
                        f"Event '{ev}' appears in explanation but not in evidence.",
                        "Remove or provide supporting evidence."
                    )
                    break

        # Intrinsic events
        for ev in sorted(expl_events & ev_events):
            for i, s in enumerate(self.sentences):
                if ev in s and ev not in ev_events:
                    add_item(
                        ErrorTypology.INTRINSIC_EVENT_ERROR,
                        i,
                        ev,
                        f"Event '{ev}' misrepresented compared to evidence.",
                        "Verify the correct event from evidence."
                    )
                    break

        # Overgeneralization
        abs_re = re.compile(r"\b(always|never|all|everyone|nobody|every|none)\b", re.IGNORECASE)
        for i, s in enumerate(self.sentences):
            if abs_re.search(s):
                corrected = re.sub(abs_re, "may", s)
                add_item(
                    ErrorTypology.OVERGENERALIZATION_ERROR,
                    i,
                    "contains absolute language",
                    "Absolute terms may overgeneralize beyond evidence.",
                    "Avoid absolute terms unless explicitly supported.",
                    corrected=corrected
                )

        # Irrelevant evidence
        ev_words = set(self._content_words(self.evidence))
        for i, s in enumerate(self.sentences):
            words = self._content_words(s)
            if not words:
                continue
            missing = [w for w in words if w not in ev_words]
            if len(missing) > max(3, len(words) * 0.5):
                add_item(
                    ErrorTypology.IRRELEVANT_EVIDENCE_ERROR,
                    i,
                    ", ".join(missing[:6]),
                    "Many terms not present in evidence; may be unsupported.",
                    "Keep explanation focused on evidence-supported facts."
                )

        # Reasoning coherence
        neg_pattern = re.compile(r"couldn\'t find|no credible evidence|unable to find|could not find", re.IGNORECASE)
        pos_pattern = re.compile(r"found|evidence shows|confirmed|verified", re.IGNORECASE)
        neg_idxs = [i for i, s in enumerate(self.sentences) if neg_pattern.search(s)]
        pos_idxs = [i for i, s in enumerate(self.sentences) if pos_pattern.search(s)]
        if neg_idxs and pos_idxs:
            for ni in neg_idxs:
                for pi in pos_idxs:
                    add_item(
                        ErrorTypology.REASONING_COHERENCE_ERROR,
                        ni,
                        "contradictory findings",
                        "Explanation contains contradictory statements about evidence found.",
                        "Resolve contradictions by clarifying the search scope and evidence.",
                    )
                    break

        if not feedback:
            feedback.append({
                "error_type": "no_error_detected",
                "sentence_index": -1,
                "sentence": "",
                "excerpt": "",
                "description": "No issues detected.",
                "suggestion": "Consider free-form reviewer if problems remain."
            })

        return feedback


if __name__ == "__main__":
    # manual testing
    explanation = (
        "... Foxworthy is famous for his 'You might be a redneck if' jokes, but searching online, we couldn't find any credible evidence that he penned this list that touches on poetry, art and literature. "
        "Rather, we found users on web forums crediting someone named Fritz Edmunds with the list. Snopes noted that 'the original compiler of this appears to be Fritz Edmunds'."
    )
    evidence = (
        "Fritz Edmunds posted to his Politically True blog on Feb. 3, 2013. Snopes checked the origin and credited Edmunds as the compiler."
    )

    d1 = Debater1(explanation, evidence)
    for item in d1.analyze():
        print("---")
        for k, v in item.items():
            print(f"{k}: {v}")
