import numpy as np
from clients import ModelClient, TransformersLMClient, LlamaCppClient
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

def parse_boolean_answer(response: str) -> bool | None:
  lower = response.lower()
  has_true = 'true' in lower
  has_false = 'false' in lower

  if has_true and has_false:
    return None
  if not has_true and not has_false:
    return None
  return has_true

def parse_rating_answer(answer: str) -> str | None:
  answer = answer.lower()

  # Handles differences between RAGAR prompt and FEVER dataset
  # Ideally would handle this in preprocessing
  # Also this probably causes collisions
  if 'support' in answer:
    return "supports"
  if 'refute' in answer:
    return "refutes"
  if 'fail' in answer:
    return "not enough info"

  print(f"\nModel gave invalid verification answer: {answer}\n")
  return None

def verify_claim(client: ModelClient, claim: str, max_iters: int = 3) -> str | None:
    question = client.send_prompt("initial_question", [claim])
    qa_pairs = []

    # Generate QA, check termination, repeat
    for _ in range(max_iters):
        # TODO: real pipeline should use RAG here
        answer = client.send_prompt("generate-answer", [question])  
        qa_pairs.append((question, answer))

        done = parse_boolean_answer(client.send_prompt("follow_up_check", [claim, qa_pairs]))
        if done == "True" or done == None:
            break

        question = client.send_prompt("follow_up_question", [claim, qa_pairs])

    # Final verification
    return parse_rating_answer(client.send_prompt("verification", [claim, qa_pairs]))

if __name__ == "__main__":
    # Download FEVER dataset https://fever.ai/dataset/fever.html
    ds = load_dataset("fever", "v1.0", trust_remote_code=True) 

    # Assumes model is downloaded and LCPP server is running
    # E.g. llama-server --reasoning-budget 0 --port 4568 -t 8 -m models\qwen2.5-1.5b-instruct-q4_k_m.gguf
    client = LlamaCppClient()
    client.register_prompt("initial_question", "prompts/ragar/initial-question.txt")
    client.register_prompt("follow_up_check", "prompts/ragar/follow-up-check.txt")
    client.register_prompt("follow_up_question", "prompts/ragar/follow-up-question.txt")
    client.register_prompt("verification", "prompts/ragar/verification.txt")
    client.register_prompt("generate-answer", "prompts/ragar/generate-answer.txt")

    # Test run on FEVER subset
    num_samples = 10
    split = ds["labelled_dev"].select(range(num_samples))
    times = []
    preds = []
    labels = []

    for i in tqdm(range(len(split))):
        claim = split[i]["claim"]
        label = split[i]["label"].lower()
        rating = verify_claim(client, claim, max_iters=3)
        labels.append(label)
        preds.append(rating)

    accuracy = sum(pred == label for pred, label in zip(preds, labels)) / num_samples
    pred_counts = Counter(preds)
    label_counts = Counter(labels)

    print()
    print(f"Accuracy: {accuracy:.3f}")
    print(pred_counts)
    print(label_counts)