"""
Copyright © 2025 bdunahu
Copyright © 2025 Ria
Copyright © 2025 Eric

You should have received a copy of the MIT license along with this file.
If not, see https://mit-license.org/
"""

from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict
import json
from .LLMClients import ModelClient
import numpy as np
import textwrap
import time


def parse_conclusivity(response: str) -> bool | None:
    """
    Given RESPONSE, attempts to parse a binary value by searching the string for keywords 'Conclusive' or 'Inconclusive'. Returns None if both or none are found.
    """
    lower = response.lower()

    has_conclusive = 'conclusive' in lower and 'inconclusive' not in lower
    has_inconclusive = 'inconclusive' in lower

    if has_conclusive and not has_inconclusive:
        return True
    elif has_inconclusive and not has_conclusive:
        return False
    return None


def parse_boolean_answer(response: str) -> bool | None:
    """
    Given RESPONSE, attempts to parse a binary value by searching the string for keywords 'True' or 'False'. Returns None if both or none are found.
    """
    lower = response.lower()
    has_true = 'true' in lower
    has_false = 'false' in lower

    if has_true and has_false:
        return None
    if not has_true and not has_false:
        return None
    return has_true


def verify_claim(client: ModelClient, claim: str, max_iters: int = 3) -> str | None:
    qa_pairs = []
    question = client.send_prompt("initial_question_agent", [claim])

    for _ in range(max_iters):
        # TODO add the rag stage
        answer = client.send_prompt("answering_agent", [question, "No evidence, try your best!"]).strip()
        qa_pairs.append((question, answer))

        done = parse_conclusivity(
            client.send_prompt("early_exit_agent", [claim, qa_pairs])
        )
        if done or done == None:
            break

        question = client.send_prompt("new_question_agent", [claim, qa_pairs]).strip()

    verdict_raw = client.send_prompt("verdict_agent", [claim, qa_pairs]).strip()
    verdict_bool = parse_boolean_answer(verdict_raw)

    result = {
        "claim": claim,
        "qa_pairs": qa_pairs,
        "verdict_raw": verdict_raw,
        "verdict_bool": verdict_bool,
    }

    return result
