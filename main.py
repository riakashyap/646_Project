from datasets import load_dataset
from src.PipelineHelper import verify_claim
from src.LLMClients import LlamaCppClient
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


def get_pred(verdict: bool):
    if verdict is None:
        pred = "not enough info"
    if verdict:
        pred = "supports"
    else:
        pred = "refutes"

    return pred


if __name__ == "__main__":
    # Download FEVER dataset https://fever.ai/dataset/fever.html
    ds = load_dataset("fever", "v1.0", trust_remote_code=True)

    # Assumes model is downloaded and LCPP server is running on port 4568
    # E.g. llama-server --reasoning-budget 0 --port 4568 -t 8 -m /path/to/model.gguf
    client = LlamaCppClient()

    # Test run on FEVER subset
    num_samples = 5
    split = ds["labelled_dev"].select(range(num_samples))
    labels = []
    preds = []
    results = []

    for i in tqdm(range(len(split))):
        claim = split[i]["claim"]
        label = split[i]["label"].lower()

        results = verify_claim(client, claim, max_iters=3)
        pred = get_pred(results["verdict_bool"])
        results["correct"] = pred == label

        preds.append(pred)
        labels.append(label)
        results.append(results)

    accuracy = sum(pred == label for pred, label in zip(preds, labels)) / num_samples
    pred_counts = Counter(preds)
    label_counts = Counter(labels)

    print()
    import pprint
    pprint.pprint(results)
    print()
    print(f"Accuracy: {accuracy:.3f}")
    print(pred_counts)
    print(label_counts)

# Local Variables:
# compile-command: "guix shell -m manifest.scm -- python3 ./main.py"
# End:
