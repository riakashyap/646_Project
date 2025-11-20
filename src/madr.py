from .model_clients import ModelClient
from .parsers import parse_boolean

def run_madr(mc: ModelClient, explanation: str, claim: str, qa_pairs: list[tuple[str, str]], max_iters: int = 3):
    f1 = mc.send_prompt("init_feedback1", [explanation, claim, qa_pairs])
    f2 = mc.send_prompt("init_feedback2", [explanation, claim, qa_pairs])

    for i in range(max_iters):
        judgement =  mc.send_prompt("judge_feedback", [f1, f2])
        if parse_boolean(judgement):
            break
        
        if i != max_iters - 1:
            f1 = mc.send_prompt("cross_feedback", [f1, f2])
            f2 = mc.send_prompt("cross_feedback", [f2, f1])

    revised_verdict = mc.send_prompt("revise_stop_check", [explanation, f1, f2])
    return revised_verdict
