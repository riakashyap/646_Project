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
from pyserini.search.lucene import LuceneSearcher
from src.LLMClients import LlamaCppClient
from src.PipelineHelper import verify_claim
from src.Retrieval import INDEX_DIR
from tqdm import tqdm
import argparse


def get_pred(verdict: bool):
    if verdict is None:
        pred = "not enough info"
    elif verdict:
        pred = "supports"
    else:
        pred = "refutes"

    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='%(prog)s [args] -- prog'
    )

    parser.add_argument('-t', '--no_think',
                        help='Whether the model should think before answering. Affects runtime. Only works on Qwen models.',
                        action='store_false')
    parser.add_argument('-r', '--ragar',
                        help='Use the original RAGAR prompts.',
                        action='store_true')
    args = parser.parse_args()

    # Download FEVER dataset https://fever.ai/dataset/fever.html
    ds = load_dataset("fever", "v1.0", trust_remote_code=True)

    # Assumes model is downloaded and LCPP server is running on port 4568
    # E.g. llama-server --reasoning-budget 0 --port 4568 -t 8 -m /path/to/model.gguf
    client = LlamaCppClient(should_think=args.no_think, use_ragar=args.ragar)

    # search index
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(1.2, 0.75)

    # Test run on FEVER subset
    num_samples = 10
    split = ds["labelled_dev"].select(range(num_samples))
    labels = []
    preds = []
    results = []

    for i in tqdm(range(len(split))):
        claim = split[i]["claim"]
        label = split[i]["label"].lower()

        result = verify_claim(client, searcher, claim, max_iters=3)
        pred = get_pred(result["verdict_bool"])
        result["correct"] = (pred == label)

        preds.append(pred)
        labels.append(label)
        results.append(result)

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
