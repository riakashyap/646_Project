def parse_boolean(text: str) -> bool:
    lower = text.lower()
    has_true = "true" in lower
    has_false = "false" in lower
    return has_true and not has_false

def parse_ternary(text: str) -> int:
    lower = text.lower()
    if "false" in lower:
        verdict = 0
    elif "true" in lower:
        verdict = 1
    elif "inconclusive" in lower:
        verdict = 2
    return verdict

def parse_conclusive(text: str) -> bool:
    lower = text.lower()
    has_inconclusive = "inconclusive" in lower
    has_conclusive = "conclusive" in lower and not has_inconclusive
    return has_conclusive and not has_inconclusive