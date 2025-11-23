"""
Copyright:

  Copyright © 2025 Eric

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file contains the MADR debate agent run-loop.

Code:
"""

from .model_clients import ModelClient
from .parsers import parse_boolean
from . import config

def run_madr(mc: ModelClient, claim: str, qa_pairs: list[tuple[str, str]], explanation: str, max_iters: int = 3):
    config.LOGGER and config.LOGGER.info(f"\n{'-' * 20}\nStarting MADR routine")
    f1 = mc.send_prompt("madr_init_fb1", [claim, qa_pairs, explanation])
    f2 = mc.send_prompt("madr_init_fb2", [claim, qa_pairs, explanation])

    for i in range(max_iters):
        config.LOGGER and config.LOGGER.info(f"Beginning MADR round {i}")

        judgement =  mc.send_prompt("madr_judge", [f1, f2])
        if parse_boolean(judgement):
            break

        if i != max_iters - 1:
            f1 = mc.send_prompt("madr_cross_fb", [f1, f2])
            f2 = mc.send_prompt("madr_cross_fb", [f2, f1])

    revised_verdict = mc.send_prompt("madr_revise", [f1, f2, explanation])
    config.LOGGER and config.LOGGER.info(f"Ending MADR routine\n{'-' * 20}\n")
    return revised_verdict
