"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric
  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file exercises the full RAGAR pipeline. Do not import this file.

Code:
"""

from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from src.model_clients import LlamaCppClient
from src.ragar_corag import RagarCorag

# Test run on FEVER subset
if __name__ == "__main__":
    num_samples = 2
    ds = load_dataset("fever", "v1.0", trust_remote_code=True)
    split = ds["labelled_dev"].select(range(num_samples))
    fever_labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH EVIDENCE"]

    # Setup CoRAG system here
    mc = LlamaCppClient("prompts/ragar")
    # mc = LlamaCppClient("prompts/custom/user", "prompts/custom/system")
    corag = RagarCorag(mc)

    labels = []
    preds = []
    outputs = []
    for i in tqdm(range(len(split))):
        claim = split[i]["claim"]
        label = split[i]["label"]

        result = corag.run(claim)
        verdict = result["verdict"]
        pred = None if verdict is None else fever_labels[verdict]

        preds.append(pred)
        labels.append(label)
        outputs.append(result)

    accuracy = sum(pred == label for pred, label in zip(preds, labels)) / num_samples
    pred_counts = Counter(preds)
    label_counts = Counter(labels)

    print()
    print(outputs)
    print(f"Accuracy: {accuracy:.3f}")
    print("Pred labels:", pred_counts)
    print("True labels:", label_counts)

# Local Variables:
# compile-command: "guix shell -m manifest.scm -- python3 ./main.py"
# End: