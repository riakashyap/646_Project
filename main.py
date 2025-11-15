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
from src.madr_corag import MadrCorag
from src.config import PROMPTS_DIR
from tqdm import tqdm
import argparse
import time
from datasets import load_dataset, Dataset, concatenate_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='%(prog)s [args] -- prog'
    )

    parser.add_argument('-t', '--think',
                        help='Whether the Qwen model should think before '
                        'answering. Affects runtime.',
                        action='store_true')
    parser.add_argument('-r', '--ragar',
                        help='Use the original RAGAR prompts.',
                        action='store_true')
    parser.add_argument('-m', '--madr',
                        help='Use Multi-Agent Debate Refinement instead of baseline.',
                        action='store_true')
    parser.add_argument('--num-agents',
                        help='Number of agents for MADR (default: 3).',
                        metavar='',
                        type=int,
                        default=3)
    parser.add_argument('--debate-rounds',
                        help='Maximum debate rounds for MADR (default: 3).',
                        metavar='',
                        type=int,
                        default=3)
    parser.add_argument('-n', '--num-claims',
                        help='The number of claims to process.',
                        metavar='',
                        type=int,
                        default=100)
    args = parser.parse_args()

    fever_labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
    ds = load_dataset("fever", "v1.0", trust_remote_code=True)

    # Split into unique 50 'REFUTES' and 50 'SUPPORTS'
    split = ds["train"]
    split = Dataset.from_pandas(split.to_pandas().drop_duplicates(subset="claim"))
    supports = split.filter(lambda row: row["label"] == "SUPPORTS").select(range(int(args.num_claims / 2)))
    refutes = split.filter(lambda row: row["label"] == "REFUTES").select(range(int(args.num_claims / 2)))
    split = concatenate_datasets([supports, refutes])
    
    # Setup prompts directory based on mode
    if args.madr:
        # MADR mode uses its own prompts
        user_prompts_dir = PROMPTS_DIR / "madr"
        sys_prompts_dir = None
    elif args.ragar:
        # Original RAGAR prompts
        user_prompts_dir = PROMPTS_DIR / "ragar"
        sys_prompts_dir = None
    else:
        # Custom prompts with system messages
        user_prompts_dir = PROMPTS_DIR / "custom" / "user"
        sys_prompts_dir = PROMPTS_DIR / "custom" / "system"

    # Setup CoRAG system here
    mc = LlamaCppClient(user_prompts_dir,
                        sys_prompts_dir,
                        think_mode_bool=args.think)

    if args.madr:
        print(f"Using MADR with {args.num_agents} agents, {args.debate_rounds} max rounds")
        corag = MadrCorag(mc, num_agents=args.num_agents, max_debate_rounds=args.debate_rounds)
    else:
        corag = RagarCorag(mc)

    labels = []
    preds = []
    outputs = []
    start = time.time()
    for i in tqdm(range(len(split))):
        claim = split[i]["claim"]
        label = split[i]["label"]

        result = corag.run(claim)
        verdict = result["verdict"]
        pred = None if verdict is None else fever_labels[verdict]

        preds.append(pred)
        labels.append(label)
        outputs.append(result)

    elapsed = time.time() - start
    accuracy = sum(pred == label for pred, label in zip(preds, labels)) / \
        args.num_claims

    print()
    # pprint(outputs)
    print(f"Accuracy: {accuracy:.8f}")
    print("Pred labels:", preds)
    print("True labels:", labels)
    print(f"Time: {elapsed:8f}")

# Local Variables:
# compile-command: "guix shell -m manifest.scm -- python3 ./main.py"
# End:
