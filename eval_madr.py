#!/usr/bin/env python3
"""
Copyright:

  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file exercises the MADR pipeline, keeping CORAG consistant. Do not import this file.

Code:
"""

from src import config
from src.madr import run_madr
from src.utils import parse_ternary, get_prompt_files, compute_metrics
from src.model_clients import LlamaCppClient
from src.ragar_corag import RagarCorag
from tqdm import tqdm
import argparse
import json
import time

def parse_arguments():
    parser = argparse.ArgumentParser(
        usage='%(prog)s [args] -- prog'
    )

    parser.add_argument('-m', '--madr-orig',
                        help='Use the original MADR prompts. This option does nothing without including --debate-stop or --debate-verdict.',
                        action='store_true')
    parser.add_argument('--debate-verdict',
                        help='Refines the verdict agent with MADR. Overrides -r.',
                        action='store_true')
    parser.add_argument('-f', '--filename',
                        help='The filename containing a CoRAG execution.',
                        metavar='',
                        type=str,
                        default="./logs/results_custom_no-think_100claims.json")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    madr_dir = config.MADR_ORIG_DIR if args.madr_orig else config.MADR_DIR
    prompt_files = get_prompt_files(madr_dir)

    fever_labels = {0: "REFUTES", 1: "SUPPORTS", 2: "NOT ENOUGH INFO", 3: "NONE"}
    mc = LlamaCppClient(prompt_files, False)

    results = None
    with open(args.filename, 'r') as f:
        results = json.load(f)["results"]

    golds = []
    outputs = []
    start = time.time()
    for i in tqdm(range(len(results))):
        result = results[i]
        claim = result["claim"]
        qa_pairs = result["qa_pairs"]
        verdict_raw = result["verdict_raw"]
        if args.debate_verdict:
            verdict_raw = run_madr(mc, claim, qa_pairs, verdict_raw)
        golds.append(result["true_label"])
        outputs.append(parse_ternary(verdict_raw))

    elapsed = time.time() - start
    preds = [fever_labels.get(output, None) for output in outputs]
    iters = [0] * len(outputs)

    metrics = compute_metrics(elapsed, iters, preds, golds)
    print(json.dumps(metrics, indent=4))

# Local Variables:
# compile-command: "guix shell -m manifest.scm -- python3 ./eval_madr.py"
# End:
