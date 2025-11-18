class ErrorTypology:
    """
    Defines the typology of errors that Debater 1 uses to categorize issues in explanations.
    """

    INTRINSIC_ENTITY_ERROR = "The generated explanation misrepresents named entities, quantities, dates, or other surface realizations from the given source."
    EXTRINSIC_ENTITY_ERROR = "The generated explanation includes new entities that are not present in the given source."
    INTRINSIC_EVENT_ERROR = "The generated explanation misrepresents events mentioned in the source."
    EXTRINSIC_EVENT_ERROR = "The generated explanation includes new events that are not present in the given source."
    INTRINSIC_NOUN_PHRASE_ERROR = "The explanation mistakenly represents the noun phrases in the given source."
    EXTRINSIC_NOUN_PHRASE_ERROR = "The explanation mistakenly represents new noun phrases that are not present in the given source."
    REASONING_COHERENCE_ERROR = "There are logical flaws in the flow of reasoning within the generated explanation."
    OVERGENERALIZATION_ERROR = "The generated explanation makes sweeping statements or draws conclusions that go beyond the evidence provided."
    IRRELEVANT_EVIDENCE_ERROR = "The generated explanation includes evidence that is not directly related to the claim."