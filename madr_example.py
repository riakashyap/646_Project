#!/usr/bin/env python3
"""
Copyright:

  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  Example script demonstrating Multi-Agent Debate Refinement (MADR) for claim verification.
  This script shows how MADR enhances the baseline CoRAG system through multi-agent debate.

Usage:
  python madr_example.py --claim "Your claim here"
  python madr_example.py --num-agents 5 --debate-rounds 4

Code:
"""

from src.model_clients import LlamaCppClient
from src.madr_corag import MadrCorag
from src.ragar_corag import RagarCorag
from src.config import PROMPTS_DIR
import argparse
import json
from pprint import pprint


def compare_baseline_vs_madr(claim: str, baseline_mc: LlamaCppClient, madr_mc: LlamaCppClient,
                              num_agents: int = 3, debate_rounds: int = 3):
    """
    Run both baseline and MADR on the same claim to compare results.
    """
    print("=" * 80)
    print("BASELINE vs MADR COMPARISON")
    print("=" * 80)
    print(f"\nClaim: {claim}\n")
    
    # Baseline
    print("\n" + "-" * 80)
    print("RUNNING BASELINE (Single Agent)...")
    print("-" * 80)
    baseline = RagarCorag(baseline_mc)
    baseline_result = baseline.run(claim)
    
    print(f"\nBaseline Verdict: {['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO'][baseline_result['verdict']]}")
    print(f"QA Pairs: {len(baseline_result['qa_pairs'])}")
    
    # MADR
    print("\n" + "-" * 80)
    print(f"RUNNING MADR ({num_agents} Agents, {debate_rounds} Rounds)...")
    print("-" * 80)
    madr = MadrCorag(madr_mc, num_agents=num_agents, max_debate_rounds=debate_rounds)
    madr_result = madr.run(claim)
    
    print(f"\nMADR Verdict: {['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO'][madr_result['verdict']]}")
    print(f"QA Pairs: {len(madr_result['qa_pairs'])}")
    
    if 'madr_metadata' in madr_result:
        metadata = madr_result['madr_metadata']
        print(f"\nDebate Statistics:")
        print(f"  Rounds completed: {metadata.get('debate_rounds', 'N/A')}")
        print(f"  Consensus reached: {metadata.get('consensus_reached', 'N/A')}")
        print(f"  Confidence scores: {metadata.get('confidence_weighted_scores', {})}")
        
        print(f"\nAgent Positions:")
        for agent_pos in metadata.get('agent_final_positions', []):
            print(f"  Agent {agent_pos['agent_id']}: {agent_pos['verdict']} "
                  f"(confidence: {agent_pos['confidence']})")
            print(f"    Reasoning: {agent_pos['reasoning'][:100]}...")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    baseline_label = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO'][baseline_result['verdict']]
    madr_label = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO'][madr_result['verdict']]
    
    print(f"Baseline verdict: {baseline_label}")
    print(f"MADR verdict:     {madr_label}")
    print(f"Agreement:        {'✓ YES' if baseline_label == madr_label else '✗ NO'}")
    
    return baseline_result, madr_result


def run_madr_only(claim: str, madr_mc: LlamaCppClient, 
                  num_agents: int = 3, debate_rounds: int = 3):
    """
    Run MADR only and display detailed results.
    """
    print("=" * 80)
    print("MULTI-AGENT DEBATE REFINEMENT (MADR)")
    print("=" * 80)
    print(f"\nClaim: {claim}")
    print(f"Agents: {num_agents}")
    print(f"Max debate rounds: {debate_rounds}\n")
    
    madr = MadrCorag(madr_mc, num_agents=num_agents, max_debate_rounds=debate_rounds)
    result = madr.run(claim)
    
    print("\n" + "-" * 80)
    print("EVIDENCE GATHERING")
    print("-" * 80)
    for i, (q, a) in enumerate(result['qa_pairs'], 1):
        print(f"\nQ{i}: {q}")
        print(f"A{i}: {a[:200]}{'...' if len(a) > 200 else ''}")
    
    print("\n" + "-" * 80)
    print("FINAL VERDICT")
    print("-" * 80)
    verdict_label = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO'][result['verdict']]
    print(f"\nVerdict: {verdict_label}")
    
    if 'madr_metadata' in result:
        print("\n" + "-" * 80)
        print("DEBATE DETAILS")
        print("-" * 80)
        metadata = result['madr_metadata']
        
        print(f"\nDebate rounds: {metadata.get('debate_rounds', 'N/A')}")
        print(f"Consensus reached: {metadata.get('consensus_reached', 'N/A')}")
        
        print(f"\nConfidence-Weighted Scores:")
        for verdict, score in metadata.get('confidence_weighted_scores', {}).items():
            print(f"  {verdict}: {score:.3f}")
        
        print(f"\nAgent Final Positions:")
        for agent_pos in metadata.get('agent_final_positions', []):
            print(f"\n  Agent {agent_pos['agent_id']}:")
            print(f"    Verdict: {agent_pos['verdict']}")
            print(f"    Confidence: {agent_pos['confidence']:.3f}")
            print(f"    Reasoning: {agent_pos['reasoning']}")
    
    print("\n" + "=" * 80)
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multi-Agent Debate Refinement (MADR) example',
        usage='%(prog)s [args]'
    )
    
    parser.add_argument('--claim', '-c',
                        help='Claim to verify (required if not using --compare-examples)',
                        type=str)
    parser.add_argument('--num-agents',
                        help='Number of debate agents (default: 3)',
                        type=int,
                        default=3)
    parser.add_argument('--debate-rounds',
                        help='Maximum debate rounds (default: 3)',
                        type=int,
                        default=3)
    parser.add_argument('--compare',
                        help='Compare baseline vs MADR',
                        action='store_true')
    parser.add_argument('--compare-examples',
                        help='Run comparison on example claims',
                        action='store_true')
    parser.add_argument('-t', '--think',
                        help='Enable thinking mode for LLM',
                        action='store_true')
    parser.add_argument('-r', '--ragar',
                        help='Use original RAGAR prompts',
                        action='store_true')
    
    args = parser.parse_args()
    
    # Setup model clients - baseline uses ragar/custom, MADR uses madr prompts
    if args.ragar:
        baseline_prompts_dir = PROMPTS_DIR / "ragar"
        baseline_sys_prompts_dir = None
    else:
        baseline_prompts_dir = PROMPTS_DIR / "custom" / "user"
        baseline_sys_prompts_dir = PROMPTS_DIR / "custom" / "system"
    
    madr_prompts_dir = PROMPTS_DIR / "madr"
    
    baseline_mc = LlamaCppClient(baseline_prompts_dir, baseline_sys_prompts_dir, think_mode_bool=args.think)
    madr_mc = LlamaCppClient(madr_prompts_dir, None, think_mode_bool=args.think)
    
    if args.compare_examples:
        # Run on example claims
        example_claims = [
            "The Earth is flat.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Barack Obama was born in Kenya.",
        ]
        
        for claim in example_claims:
            compare_baseline_vs_madr(claim, baseline_mc, madr_mc, args.num_agents, args.debate_rounds)
            print("\n" * 2)
    
    elif args.claim:
        if args.compare:
            compare_baseline_vs_madr(args.claim, baseline_mc, madr_mc, args.num_agents, args.debate_rounds)
        else:
            run_madr_only(args.claim, madr_mc, args.num_agents, args.debate_rounds)
    
    else:
        parser.print_help()
        print("\nError: Please provide --claim or use --compare-examples")

# Local Variables:
# compile-command: "python3 ./madr_example.py --claim 'Your claim here'"
# End:
