from .model_clients import ModelClient
from .parsers import parse_boolean

def run_madr(mc: ModelClient, explanation: str, claim: str, qa_pairs: list[tuple[str, str]], max_iters: int = 3):
    f1 = mc.send_prompt("madr_init_fb1", [explanation, claim, qa_pairs])
    f2 = mc.send_prompt("madr_init_fb2", [explanation, claim, qa_pairs])

    for i in range(max_iters):
        judgement =  mc.send_prompt("madr_judge", [f1, f2])
        if parse_boolean(judgement):
            break
        
        if i != max_iters - 1:
            f1 = mc.send_prompt("madr_cross_fb", [f1, f2])
            f2 = mc.send_prompt("madr_cross_fb", [f2, f1])

    revised_verdict = mc.send_prompt("madr_revise", [explanation, f1, f2])
    return revised_verdict
