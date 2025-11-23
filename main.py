#!/usr/bin/env python3
"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric
  Copyright © 2025 Ria
  Copyright © 2025 uchuuronin

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
from tqdm import tqdm
import argparse
import time
import os
from datasets import load_dataset, Dataset, concatenate_datasets

from src import config
from reranker import E2RankReranker

def parse_arguments():
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
    parser.add_argument('-n', '--num-claims',
                        help='The number of claims to process.',
                        metavar='',
                        type=int,
                        default=100)
    parser.add_argument('--reranker',
                        help='Enable reranking after BM25 retrieval.',
                        action='store_true')
    parser.add_argument('--debate-stop',
                        help='Refines the stop_check agent with MADR. Overrides -r.',
                        action='store_true')
    parser.add_argument('-l', '--log-trace',
                        help='Output a trace to the log file (define in config.py). Overrides --num-claims to 2.',
                        action='store_true')

    args = parser.parse_args()

    # handle overrides
    if args.log_trace:
        config.make_logger()
        args.num_claims = 2

    if args.debate_stop:
        args.ragar = False

    return args


if __name__ == "__main__":
    args = parse_arguments()

    fever_labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
    ds = load_dataset("fever", "v1.0", trust_remote_code=True)

    # Split into unique 50 'REFUTES' and 50 'SUPPORTS'
    split = ds["train"]
    split = Dataset.from_pandas(split.to_pandas().drop_duplicates(subset="claim"))
    supports = split.filter(lambda row: row["label"] == "SUPPORTS").select(range(int(args.num_claims / 2)))
    refutes = split.filter(lambda row: row["label"] == "REFUTES").select(range(int(args.num_claims / 2)))
    split = concatenate_datasets([supports, refutes])

    if args.ragar:
        config.LOGGER and config.LOGGER.info("Using RAGAR prompts.")
        prompts_dir = config.PROMPTS_DIR / "ragar"
    else:
        config.LOGGER and config.LOGGER.info("Using CUSTOM prompts.")
        prompts_dir = config.PROMPTS_DIR / "custom"

    # Initialize reranker if requested
    reranker = None
    if args.reranker:
        try:
            reranker = E2RankReranker()
        except (OSError, FileNotFoundError) as e:
            print(f"ERROR: Model files not found or download failed: {e}")
            reranker = None
    
    # Setup CoRAG system here
    mc = LlamaCppClient(prompts_dir, think_mode_bool=args.think)
    corag = RagarCorag(mc, args.debate_stop, reranker=reranker)

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
    print(f"Accuracy: {accuracy:.8f}")
    print("Pred labels:", preds)
    print("True labels:", labels)
    print(f"Time: {elapsed:8f}")

# Local Variables:
# compile-command: "guix shell -m manifest.scm -- python3 ./main.py"
# End:
