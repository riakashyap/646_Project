"""
Copyright:

  Copyright © 2025 Ananya-Jha-code
  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

  

run_trainer.py
----------------
Entry point for finetuning the reranker on FEVER.

Usage:
    python run_trainer.py --model_type pairwise
    python run_trainer.py --model_type e2rank --lr 3e-5 --epochs 3
"""

import argparse
from pathlib import Path

from .trainer import RerankerTrainer, FeverRerankDataset
from src.config import (
    CLAIMS_PATH, QRELS_PATH, RANKLISTS_PATH, INDEX_DIR
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pairwise", "e2rank"],
        default="pairwise",
        help="Which reranker architecture to train"
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["pairwise_exp", "pairwise_hinge", "listwise_ce", "pointwise_bce"],
        default="pairwise_exp",
        help="Which loss function to use"
    )

    parser.add_argument("--margin", type=float, default=1.0, help="Margin for hinge loss")
    parser.add_argument("--pos_weight", type=float, default=None, help="Positive weight for BCE")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_negatives", type=int, default=1)

    parser.add_argument(
        "--model_name",
        type=str,
        default="naver/trecdl22-crossencoder-debertav3",
        help="Base model to finetune"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/fever_finetuned",
        help="Where to save the finetuned model"
    )

    return parser.parse_args()


def main():
    args = parse_args()


    # 1. Build Dataset
    print("\n=== Building FEVER Dataset ===")
    dataset = FeverRerankDataset(
        claims_path=Path(CLAIMS_PATH),
        qrels_path=Path(QRELS_PATH),
        ranklist_path=Path(RANKLISTS_PATH),
        index_path=Path(INDEX_DIR),
        max_negatives=args.max_negatives
    )

    # 3. Initialize Trainer
    print("\n=== Initializing Trainer ===")
    trainer = RerankerTrainer(
        model_type=args.model_type,
        model_name=args.model_name,
        lr=args.lr,
        use_layerwise=(args.model_type == "e2rank")
    )

    # 4. Train
    print("\n=== Starting Training ===")
    trainer.train(
        dataset=dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )

    # 5. Save model
    print("\n=== Saving Model ===")
    trainer.save(args.save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()