"""
Copyright:

  Copyright © 2025 Eric
  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file tests the efficacy of a CoRAG pipeline by computing F1 scores from predicted and true labels.
  Assumes labels are stored in log files residing in the /logs directory.

  General approach adapted from materials presented in the course curriculum
  COMPSCI646, UMass Amherst, Fall 2025.

Code:
"""

import pathlib
import os
import ast
from sklearn.metrics import classification_report

LOGS_DIR = (pathlib.Path(__file__).parent / "../logs").resolve()

def parse_pipeline_log(path: str):
    """Extract predictions, true labels, and runtime from a log file."""

    # Parse contents
    pred_labels = true_labels = total_time = None
    iterations = []

    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Pred labels:"):
                pred_labels = ast.literal_eval(line.split("Pred labels:")[1].strip())
            elif line.startswith("True labels:"):
                true_labels = ast.literal_eval(line.split("True labels:")[1].strip())
            elif line.startswith("Time:"):
                total_time = float(line.split("Time:")[1].strip())
            elif "Iterations:" in line:    
                iterations.append(int(line.split("Iterations:")[1].strip()))


    # Computes F1 scores
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    support_f1 = report.get("SUPPORTS", {}).get("f1-score", 0)
    refute_f1 = report.get("REFUTES", {}).get("f1-score", 0)
    weighted_f1 = report["weighted avg"]["f1-score"]

    failed = len([pred for pred in pred_labels if pred not in ["SUPPORTS", "REFUTES"]])
    tpc = total_time / len(pred_labels) if len(pred_labels) > 0 else 0
    avg_iters = sum(iterations) / len(iterations) if iterations else 0

    return support_f1, refute_f1, failed, weighted_f1, tpc, avg_iters

if __name__ == "__main__":

    print(f"{'File':<30} {'SUP (F1)':>10} {'REF (F1)':>10} {'#Fail':>8} {'Wtd F1':>10} {'Time/Claim(s)':>15} {'Avg Iters':>12}")
    print("-" * 85)

    for file_name in sorted(os.listdir(LOGS_DIR )):
        if not file_name.endswith(".txt"):
            continue

        path = LOGS_DIR / file_name
        support_f1, refute_f1, failed, weighted_f1, tpc, avg_iters = parse_pipeline_log(path)
        print(f"{file_name:<30} {support_f1:>10.3f} {refute_f1:>10.3f} {failed:>8d} {weighted_f1:>10.3f} {tpc:>15.3f} {avg_iters:>12.2f}")