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
from datasets import load_dataset, Dataset, concatenate_datasets
from datetime import datetime
from pathlib import Path
from reranker import E2RankReranker
from sklearn.metrics import classification_report
from src import config
from src.model_clients import LlamaCppClient
from src.ragar_corag import RagarCorag
from src.utils import get_prompt_files
from tqdm import tqdm
import argparse
import json
import numpy as np
import time


def parse_arguments():
    parser = argparse.ArgumentParser(
        usage='%(prog)s [args] -- prog'
    )

    parser.add_argument('-t', '--think',
                        help='Whether the Qwen model should think before '
                        'answering. Affects runtime.',
                        action='store_true')
    parser.add_argument('-r', '--ragar-orig',
                        help='Use the original RAGAR prompts.',
                        action='store_true')
    parser.add_argument('-m', '--madr-orig',
                        help='Use the original MADR prompts. This option does nothing without including --debate-stop or --debate-verdict.',
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

    return args

def setup_fever(num_claims: int):
    # Split into unique half 'REFUTES' and half 'SUPPORTS'
    ds = load_dataset("fever", "v1.0", trust_remote_code=True)
    half = int(num_claims / 2)
    split = Dataset.from_pandas(ds["train"].to_pandas().drop_duplicates(subset="claim"))
    supports = split.filter(lambda row: row["label"] == "SUPPORTS").select(range(half))
    refutes = split.filter(lambda row: row["label"] == "REFUTES").select(range(half))
    return concatenate_datasets([supports, refutes])

if __name__ == "__main__":
    args = parse_arguments()

    ragar_dir = config.RAGAR_DIR
    madr_dir = config.MADR_DIR
    if args.ragar_orig:
        ragar_dir = config.RAGAR_ORIG_DIR
        config.LOGGER.info("Using original RAGAR prompts.")
    if args.madr_orig:
        madr_dir = config.MADR_ORIG_DIR
        config.LOGGER.info("Using original MADR prompts.")
    prompt_files = get_prompt_files(ragar_dir, madr_dir)

    reranker = None
    if args.reranker:
        try:
            reranker = E2RankReranker()
        except (OSError, FileNotFoundError) as e:
            print(f"ERROR: Model files not found or download failed: {e}")
            reranker = None

    fever_labels = {0: "REFUTES", 1: "SUPPORTS", 2: "NOT ENOUGH INFO"}
    fever_split = setup_fever(args.num_claims)

    # Setup CoRAG system here
    mc = LlamaCppClient(prompt_files, think_mode_bool=args.think)
    corag = RagarCorag(mc, args.debate_stop, args.debate_verdict, reranker=reranker)

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

    print(preds)

    # Compute metrics
    report = classification_report(golds, preds, output_dict=True, zero_division=0)
    metrics = {
        "time_elapsed": elapsed,
        "accuracy": sum(pred == gold for pred, gold in zip(preds, golds)) / len(preds),
        "support_preds": preds.count("SUPPORTS"),
        "refute_preds": preds.count("REFUTES"),
        "nei_preds": preds.count("NOT ENOUGH INFO"),
        "failed_preds": preds.count(None),
        "tpc": elapsed / len(preds),
        "avg_iters": np.mean(iters),
        "support_f1": report.get("SUPPORTS", {}).get("f1-score", 0),
        "refute_f1": report.get("REFUTES", {}).get("f1-score", 0),
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

    print(json.dumps(metrics, indent=4))

    # don't output if we're just logging.
    if not args.log_trace:
        flags = [
            "think_"       if args.think else "",
            "ragar_"       if args.ragar else "",
            "rerank_"      if args.reranker else "",
            "madrstop_"    if args.debate_stop else "",
            "madrverdict_" if args.debate_verdict else "",
        ]
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        to_write = str(config.LOGS_DIR / (
            f"{timestamp}--{config.EVAL_OUT_FNAME_BASE}__"
            + f"".join(flags)
            + f"{args.num_claims}.json"
        ))
        with open(to_write, "w") as file:
            json.dump(metrics, file, indent=4)
        print(f"Wrote {to_write}.")
