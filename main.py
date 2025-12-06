#!/usr/bin/env python3
BANNER = """\
    _/_/_/_/    _/_/    _/_/_/    _/        _/_/_/_/
   _/        _/    _/  _/    _/  _/        _/
  _/_/_/    _/_/_/_/  _/_/_/    _/        _/_/_/
 _/        _/    _/  _/    _/  _/        _/
_/        _/    _/  _/_/_/    _/_/_/_/  _/_/_/_/"""
VERSION = "v1.0.0"
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
from src import config
from src.model_clients import LlamaCppClient
from src.utils import get_prompt_files, compute_metrics
from tqdm import tqdm
import argparse
import json
import time

class BannerVersion(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(BANNER)
        print(f"version {VERSION}")
        parser.exit()

def parse_arguments():
    parser = argparse.ArgumentParser(
        usage='%(prog)s [args] -- prog'
    )

    parser.add_argument('-v', '--version',
                        action=BannerVersion,
                        nargs=0,
                        help="show the version number")
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
                        help='Refines the stop_check agent with MADR.',
                        action='store_true')
    parser.add_argument('--debate-verdict',
                        help='Refines the verdict agent with MADR.',
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
    # load these later to enable fast --help and --version
    from reranker import E2RankReranker, CrossEncoderReranker
    from src.ragar_corag import RagarCorag

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
            reranker = CrossEncoderReranker()
        except (OSError, FileNotFoundError) as e:
            print(f"ERROR: Model files not found or download failed: {e}")
            reranker = None

    fever_labels = {0: "REFUTES", 1: "SUPPORTS", 2: "NOT ENOUGH INFO", 3: "NONE"}
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
    metrics = compute_metrics(elapsed, iters, preds, golds)
    print(json.dumps(metrics, indent=4))

    # don't output if we're just logging.
    if not args.log_trace:
        flags = [
            "think_"       if args.think else "",
            "ragarorig_"   if args.ragar_orig else "",
            "madrorig_"    if args.madr_orig else "",
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
