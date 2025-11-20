from .model_clients import ModelClient
from .parsers import parse_boolean

def run_madr(mc: ModelClient, claim: str, qa_pairs: list[tuple[str, str]], explanation: str, max_iters: int = 3):
    f1 = mc.send_prompt("madr_init_fb1", [claim, qa_pairs, explanation])
    f2 = mc.send_prompt("madr_init_fb2", [claim, qa_pairs, explanation])

    for i in range(max_iters):
        judgement =  mc.send_prompt("madr_judge", [f1, f2])
        if parse_boolean(judgement):
            break
        
        if i != max_iters - 1:
            f1 = mc.send_prompt("madr_cross_fb", [f1, f2])
            f2 = mc.send_prompt("madr_cross_fb", [f2, f1])

    revised_verdict = mc.send_prompt("madr_revise", [f1, f2, explanation])
    return revised_verdict
