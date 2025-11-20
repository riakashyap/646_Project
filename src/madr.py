from .model_clients import ModelClient

def is_stop_judgement(judgement: str) -> bool:
    lower = judgement.lower()
    has_true = "true" in lower
    has_false = "false" in lower
    return has_true and not has_false

def run_madr(mc: ModelClient, claim: str, qa_pairs: list[tuple[str, str]], max_iters: int = 3):
    f1 = mc.send_prompt("init_feedback1", [claim, qa_pairs])
    f2 = mc.send_prompt("init_feedback2", [claim, qa_pairs])

    for i in range(max_iters):
        judgement =  mc.send_prompt("judge_feedback", [f1, f2])
        if is_stop_judgement(judgement):
            break
        
        if i != max_iters - 1:
            f1 = mc.send_prompt("cross_feedback", [f1, f2])
            f2 = mc.send_prompt("cross_feedback", [f2, f1])

    # TODO: Should we return the final feedbacks, or have a refiner that determines if we want to stop/verdict/other based on the feedbacks.
    return f1, f2
