#!/usr/bin/env python3
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
from pprint import pprint
from src.model_clients import LlamaCppClient
from src.ragar_corag import RagarCorag
from src.utils import PROMPTS_DIR
from tqdm import tqdm
import argparse

# Test run on FEVER subset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='%(prog)s [args] -- prog'
    )

    parser.add_argument('-t', '--think',
                        help='Whether the Qwen model should think before '
                        'answering. Affects runtime.',
                        action='store_false')
    parser.add_argument('-r', '--ragar',
                        help='Use the original RAGAR prompts.',
                        action='store_true')
    parser.add_argument('-n', '--num-claims',
                        help='The number of claims to process.',
                        metavar='',
                        type=int,
                        default=3)
    args = parser.parse_args()

    ds = load_dataset("fever", "v1.0", trust_remote_code=True)
    split = ds["labelled_dev"].select(range(args.num_claims))
    fever_labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]

    user_prompts_dir = PROMPTS_DIR / "ragar"
    sys_prompts_dir = None
    if not args.ragar:
        user_prompts_dir = PROMPTS_DIR / "custom" / "user"
        sys_prompts_dir = PROMPTS_DIR  / "custom" / "system"

    # Setup CoRAG system here
    mc = LlamaCppClient(user_prompts_dir,
                        sys_prompts_dir,
                        think_mode=args.think)

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

    accuracy = sum(pred == label for pred, label in zip(preds, labels)) / \
        args.num_claims
    pred_counts = Counter(preds)
    label_counts = Counter(labels)

    print()
    pprint(outputs)
    print(f"Accuracy: {accuracy:.3f}")
    print("Pred labels:", pred_counts)
    print("True labels:", label_counts)

# Local Variables:
# compile-command: "guix shell -m manifest.scm -- python3 ./main.py"
# End:
