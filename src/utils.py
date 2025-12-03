"""
Copyright:

  Copyright © 2025 Eric
  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file contains various utility files:
    - AI-response parsers
    - OS/Dir common operations

Code:
"""

import os

def parse_boolean(text: str) -> bool:
    lower = text.lower()
    has_true = "true" in lower
    has_false = "false" in lower
    return has_true and not has_false

def parse_ternary(text: str) -> int | None:
    lower = text.lower()
    verdict = None
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

def get_prompt_files(*dirs) -> list[str]:
    files = []
    for d in dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if os.path.isfile(os.path.join(d, f)):
                    files.append(os.path.join(d, f))
    return files
