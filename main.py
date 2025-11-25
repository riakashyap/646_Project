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
import json
import os
import numpy as np
from src.model_clients import LlamaCppClient
from src.ragar_corag import RagarCorag
from tqdm import tqdm
import argparse
import time
import os
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.metrics import classification_report

from src import config

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
    parser.add_argument('--debate-stop',
                        help='Refines the stop_check agent with MADR. Overrides -r.',
                        action='store_true')
    parser.add_argument('--debate-verdict',
                        help='Refines the verdict agent with MADR. Overrides -r.',
                        action='store_true')
    parser.add_argument('-l', '--log-trace',
                        help='Output a trace to the log file (define in config.py). Overrides --num-claims to 2.',
                        action='store_true')

    args = parser.parse_args()

    # handle overrides
    if args.log_trace:
        args.num_claims = 2
        config.make_log_file()
        config.LOGGER.disabled = False
        config.LOGGER.info("\n" + "═" * 40)
        config.LOGGER.info("Starting pipeline...")

    if args.debate_stop or args.debate_verdict:
        args.ragar = False

    return args

def setup_fever(num_claims: int):
    # Split into unique 50 'REFUTES' and 50 'SUPPORTS'
    ds = load_dataset("fever", "v1.0", trust_remote_code=True)
    half = int(num_claims / 2)
    split = Dataset.from_pandas(ds["train"].to_pandas().drop_duplicates(subset="claim"))
    supports = split.filter(lambda row: row["label"] == "SUPPORTS").select(range(half))
    refutes = split.filter(lambda row: row["label"] == "REFUTES").select(range(half))
    return concatenate_datasets([supports, refutes])

if __name__ == "__main__":
    args = parse_arguments()

    if args.ragar:
        config.LOGGER.info("Using RAGAR prompts.")
        prompts_dir = config.PROMPTS_DIR / "ragar"
    else:
        config.LOGGER.info("Using CUSTOM prompts.")
        prompts_dir = config.PROMPTS_DIR / "custom"

    fever_labels = {0: "REFUTES", 1: "SUPPORTS", 2: "NOT ENOUGH INFO"}
    fever_split = setup_fever(args.num_claims)

    # Setup CoRAG system here
    mc = LlamaCppClient(prompts_dir, think_mode_bool=args.think)
    corag = RagarCorag(mc, args.debate_stop, args.debate_verdict)

    # Run pipeline on claims 
    golds = []
    outputs = []
    start = time.time()
    for i in tqdm(range(len(fever_split))):
        claim = fever_split[i]["claim"]
        gold = fever_split[i]["label"]
        output = corag.run(claim)
        golds.append(gold)
        outputs.append(output)
    
    # Extract relevant data
    elapsed = time.time() - start
    preds = [fever_labels.get(output["verdict"], None) for output in outputs]
    iters = [output["iterations"] for output in outputs]

    # Compute metrics
    report = classification_report(golds, preds, output_dict=True, zero_division=0)
    metrics = {
        "time_elapsed": elapsed,
        "num_support": preds.count("SUPPORTS"),
        "num_refute": preds.count("REFUTES"),
        "num_failed": preds.count(None),
        "accuracy": sum(pred == gold for pred, gold in zip(preds, golds)) / len(preds),
        "tpc": elapsed / len(preds),
        "avg_iters": np.mean(iters),
        "support_f1": report.get("SUPPORTS", {}).get("f1-score", 0),
        "refute_f1": report.get("REFUTES", {}).get("f1-score", 0),
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

    print(json.dumps(metrics, indent=4))
    with open(f"logs/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

# Local Variables:
# compile-command: "guix shell -m manifest.scm -- python3 ./main.py"
# End: