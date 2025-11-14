#!/usr/bin/env python3
"""
Copyright:

  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  Evaluate MADR on the FEVER dataset and compare against baseline.
  Computes accuracy, precision, recall, F1 for claim verification.

Usage:
  python evaluate_madr.py --num-samples 10
  python evaluate_madr.py --num-agents 3 --debate-rounds 2 --num-samples 50

Code:
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from src.model_clients import LlamaCppClient
from src.madr_corag import MadrCorag
from src.ragar_corag import RagarCorag
from src.config import PROMPTS_DIR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from typing import List, Dict


# Label mapping
LABEL_MAP = {
    'SUPPORTS': 1,
    'REFUTES': 0,
    'NOT ENOUGH INFO': 2
}

REVERSE_LABEL_MAP = {
    0: 'REFUTES',
    1: 'SUPPORTS', 
    2: 'NOT ENOUGH INFO'
}


def load_fever_samples(num_samples: int = 10, split: str = 'labelled_dev'):
    """Load FEVER dataset samples with ground truth labels."""
    print(f"Loading {num_samples} samples from FEVER {split} set...")
    
    dataset = load_dataset("fever", "v1.0", split=split)
    
    # Sample and deduplicate by claim ID
    samples = []
    seen_ids = set()
    
    for item in dataset:
        if item['id'] not in seen_ids:
            samples.append({
                'id': item['id'],
                'claim': item['claim'],
                'label': item['label']
            })
            seen_ids.add(item['id'])
            
            if len(samples) >= num_samples:
                break
    
    return samples


def evaluate_system(samples: List[Dict], model_client: LlamaCppClient, 
                     system_name: str, num_agents: int = None, 
                     max_debate_rounds: int = None) -> Dict:
    """
    Run a verification system on samples and return predictions + metadata.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {system_name}")
    print(f"{'='*80}")
    
    if system_name == "MADR":
        system = MadrCorag(model_client, num_agents=num_agents, 
                          max_debate_rounds=max_debate_rounds)
    else:
        system = RagarCorag(model_client)
    
    predictions = []
    ground_truths = []
    metadata_list = []
    
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Processing claim {sample['id']}: {sample['claim'][:60]}...")
        
        try:
            result = system.run(sample['claim'])
            
            pred_label = result['verdict']
            true_label = LABEL_MAP[sample['label']]
            
            predictions.append(pred_label)
            ground_truths.append(true_label)
            
            meta = {
                'claim_id': sample['id'],
                'claim': sample['claim'],
                'predicted': REVERSE_LABEL_MAP[pred_label],
                'ground_truth': sample['label'],
                'correct': pred_label == true_label
            }
            
            if 'madr_metadata' in result:
                meta['madr_metadata'] = result['madr_metadata']
            
            metadata_list.append(meta)
            
            print(f"  Predicted: {REVERSE_LABEL_MAP[pred_label]} | "
                  f"Ground Truth: {sample['label']} | "
                  f"{'CORRECT' if pred_label == true_label else 'WRONG'}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            # Use "NOT ENOUGH INFO" as fallback
            predictions.append(2)
            ground_truths.append(LABEL_MAP[sample['label']])
            metadata_list.append({
                'claim_id': sample['id'],
                'claim': sample['claim'],
                'predicted': 'ERROR',
                'ground_truth': sample['label'],
                'correct': False,
                'error': str(e)
            })
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    accuracy = accuracy_score(ground_truths, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truths, predictions, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(ground_truths, predictions, 
                                       average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(ground_truths, predictions, labels=[0, 1, 2])
    
    results = {
        'system_name': system_name,
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_per_class': {
            'REFUTES': precision_per_class[0],
            'SUPPORTS': precision_per_class[1],
            'NOT ENOUGH INFO': precision_per_class[2]
        },
        'recall_per_class': {
            'REFUTES': recall_per_class[0],
            'SUPPORTS': recall_per_class[1],
            'NOT ENOUGH INFO': recall_per_class[2]
        },
        'f1_per_class': {
            'REFUTES': f1_per_class[0],
            'SUPPORTS': f1_per_class[1],
            'NOT ENOUGH INFO': f1_per_class[2]
        },
        'support_per_class': {
            'REFUTES': int(support_per_class[0]),
            'SUPPORTS': int(support_per_class[1]),
            'NOT ENOUGH INFO': int(support_per_class[2])
        },
        'confusion_matrix': cm.tolist(),
        'elapsed_time': elapsed,
        'avg_time_per_sample': elapsed / len(samples),
        'metadata': metadata_list
    }
    
    return results


def print_results(results: Dict):
    """Print evaluation results in a readable format."""
    print(f"\n{'='*80}")
    print(f"RESULTS: {results['system_name']}")
    print(f"{'='*80}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.3f}")
    print(f"  Precision: {results['precision_macro']:.3f} (macro)")
    print(f"  Recall:    {results['recall_macro']:.3f} (macro)")
    print(f"  F1 Score:  {results['f1_macro']:.3f} (macro)")
    
    print(f"\nPer-Class Metrics:")
    for label in ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO']:
        print(f"\n  {label}:")
        print(f"    Precision: {results['precision_per_class'][label]:.3f}")
        print(f"    Recall:    {results['recall_per_class'][label]:.3f}")
        print(f"    F1 Score:  {results['f1_per_class'][label]:.3f}")
        print(f"    Support:   {results['support_per_class'][label]}")
    
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"             REF  SUP  NEI")
    cm = results['confusion_matrix']
    labels_short = ['REFUTES', 'SUPPORTS', 'NEI']
    for i, true_label in enumerate(['REFUTES', 'SUPPORTS', 'NEI']):
        print(f"  True {true_label:8s} {cm[i][0]:3d}  {cm[i][1]:3d}  {cm[i][2]:3d}")
    
    print(f"\nTime:")
    print(f"  Total:     {results['elapsed_time']:.1f}s")
    print(f"  Per claim: {results['avg_time_per_sample']:.1f}s")


def compare_results(baseline_results: Dict, madr_results: Dict):
    """Compare baseline vs MADR results."""
    print(f"\n{'='*80}")
    print(f"BASELINE vs MADR COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nOverall Performance:")
    print(f"  Metric           Baseline   MADR      Delta")
    print(f"  {'-'*50}")
    
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for metric, name in zip(metrics, metric_names):
        base_val = baseline_results[metric]
        madr_val = madr_results[metric]
        delta = madr_val - base_val
        delta_pct = (delta / base_val * 100) if base_val > 0 else 0
        
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        direction = "[UP]" if delta > 0 else "[DOWN]" if delta < 0 else "[SAME]"
        
        print(f"  {name:12s}   {base_val:.3f}    {madr_val:.3f}   "
              f"{delta_str} ({delta_pct:+.1f}%) {direction}")
    
    print(f"\nEfficiency:")
    base_time = baseline_results['avg_time_per_sample']
    madr_time = madr_results['avg_time_per_sample']
    time_ratio = madr_time / base_time if base_time > 0 else 0
    print(f"  Baseline: {base_time:.1f}s per claim")
    print(f"  MADR:     {madr_time:.1f}s per claim ({time_ratio:.1f}x slower)")
    
    # Agreement analysis
    base_meta = baseline_results['metadata']
    madr_meta = madr_results['metadata']
    
    agreements = sum(1 for b, m in zip(base_meta, madr_meta) 
                    if b['predicted'] == m['predicted'])
    agreement_rate = agreements / len(base_meta) if base_meta else 0
    
    print(f"\nAgreement:")
    print(f"  Systems agree on {agreements}/{len(base_meta)} claims ({agreement_rate:.1%})")
    
    # Cases where MADR corrected baseline errors
    baseline_wrong_madr_right = sum(
        1 for b, m in zip(base_meta, madr_meta)
        if not b['correct'] and m['correct']
    )
    
    madr_wrong_baseline_right = sum(
        1 for b, m in zip(base_meta, madr_meta)
        if b['correct'] and not m['correct']
    )
    
    print(f"\nError Correction:")
    print(f"  MADR fixed baseline errors:  {baseline_wrong_madr_right}")
    print(f"  MADR introduced new errors:  {madr_wrong_baseline_right}")
    print(f"  Net improvement:             {baseline_wrong_madr_right - madr_wrong_baseline_right}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MADR on FEVER dataset'
    )
    
    parser.add_argument('--num-samples', '-n',
                       type=int,
                       default=10,
                       help='Number of samples to evaluate (default: 10)')
    parser.add_argument('--num-agents',
                       type=int,
                       default=3,
                       help='Number of MADR agents (default: 3)')
    parser.add_argument('--debate-rounds',
                       type=int,
                       default=3,
                       help='Max MADR debate rounds (default: 3)')
    parser.add_argument('--baseline-only',
                       action='store_true',
                       help='Evaluate baseline only')
    parser.add_argument('--madr-only',
                       action='store_true',
                       help='Evaluate MADR only')
    parser.add_argument('--output', '-o',
                       type=str,
                       help='Save results to JSON file')
    parser.add_argument('--split',
                       type=str,
                       default='labelled_dev',
                       choices=['train', 'labelled_dev', 'paper_dev', 'paper_test'],
                       help='FEVER dataset split (default: labelled_dev)')
    parser.add_argument('-t', '--think',
                       action='store_true',
                       help='Enable thinking mode for LLM')
    
    args = parser.parse_args()
    
    # Load samples
    samples = load_fever_samples(args.num_samples, args.split)
    print(f"Loaded {len(samples)} samples")
    
    # Setup model clients
    baseline_prompts_dir = PROMPTS_DIR / "custom" / "user"
    baseline_sys_prompts_dir = PROMPTS_DIR / "custom" / "system"
    madr_prompts_dir = PROMPTS_DIR / "madr"
    
    results = {}
    
    # Evaluate baseline
    if not args.madr_only:
        baseline_mc = LlamaCppClient(baseline_prompts_dir, baseline_sys_prompts_dir,
                                     think_mode_bool=args.think)
        results['baseline'] = evaluate_system(samples, baseline_mc, "Baseline")
        print_results(results['baseline'])
    
    # Evaluate MADR
    if not args.baseline_only:
        madr_mc = LlamaCppClient(madr_prompts_dir, None, think_mode_bool=args.think)
        results['madr'] = evaluate_system(samples, madr_mc, "MADR",
                                         num_agents=args.num_agents,
                                         max_debate_rounds=args.debate_rounds)
        print_results(results['madr'])
    
    # Comparison
    if 'baseline' in results and 'madr' in results:
        compare_results(results['baseline'], results['madr'])
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

# Local Variables:
# compile-command: "python3 ./evaluate_madr.py --num-samples 10"
# End:
