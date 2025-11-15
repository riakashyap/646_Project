"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric
  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file implements Multi-Agent Debate Refinement (MADR) for claim verification.
  Multiple LLM agents independently analyze claims, debate verdicts, and refine
  their positions through iterative rounds until consensus is reached.

Code:
"""

from .ragar_corag import RagarCorag
from .model_clients import ModelClient
from .config import INDEX_DIR
from pyserini.search.lucene import LuceneSearcher
from typing import List, Tuple, Dict, Optional
import json


class MadrAgent:
    """Represents a single debate agent with its own reasoning and verdict."""
    
    def __init__(self, agent_id: int, mc: ModelClient):
        self.agent_id = agent_id
        self._mc = mc
        self.current_verdict: Optional[int] = None
        self.current_reasoning: str = ""
        self.confidence: float = 0.0
    
    def analyze_claim(self, claim: str, evidence: str, other_verdicts: List[Dict] = None) -> Tuple[int, str, float]:
        """
        Analyze claim with evidence and optionally consider other agents' verdicts.
        
        Returns:
            (verdict, reasoning, confidence) where verdict is 0=REFUTES, 1=SUPPORTS, 2=NOT_ENOUGH_INFO
        """
        if other_verdicts is None or len(other_verdicts) == 0:
            # Initial analysis - use agent_initial_analysis prompt
            response = self._mc.send_prompt("agent_initial_analysis", [
                str(self.agent_id),
                claim,
                evidence
            ])
        else:
            # Refinement round - use agent_debate_round prompt
            other_positions = "\n\n".join([
                f"Agent {v['agent_id']} (Confidence: {v['confidence']:.2f}):\n"
                f"  Verdict: {['REFUTES', 'SUPPORTS', 'NOT_ENOUGH_INFO'][v['verdict']]}\n"
                f"  Reasoning: {v['reasoning']}"
                for v in other_verdicts if v['agent_id'] != self.agent_id
            ])
            
            response = self._mc.send_prompt("agent_debate_round", [
                str(self.agent_id),
                claim,
                evidence,
                ['REFUTES', 'SUPPORTS', 'NOT_ENOUGH_INFO'][self.current_verdict] if self.current_verdict is not None else 'None',
                self.current_reasoning,
                f"{self.confidence:.2f}",
                other_positions
            ])
        
        verdict, reasoning, confidence = self._parse_agent_response(response)
        
        self.current_verdict = verdict
        self.current_reasoning = reasoning
        self.confidence = confidence
        
        return verdict, reasoning, confidence
    
    def _parse_agent_response(self, response: str) -> Tuple[int, str, float]:
        """Parse agent response to extract verdict, reasoning, and confidence."""
        lines = response.strip().split('\n')
        verdict = 2  # default to NOT_ENOUGH_INFO
        reasoning = ""
        confidence = 0.5
        
        for line in lines:
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict_str = line.replace("VERDICT:", "").strip().upper()
                if "REFUTE" in verdict_str or "FALSE" in verdict_str:
                    verdict = 0
                elif "SUPPORT" in verdict_str or "TRUE" in verdict_str:
                    verdict = 1
                else:
                    verdict = 2
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
                except:
                    confidence = 0.5
        
        # If reasoning is multi-line, capture it all
        if "REASONING:" in response:
            reasoning_start = response.find("REASONING:") + len("REASONING:")
            reasoning_end = response.find("CONFIDENCE:", reasoning_start)
            if reasoning_end == -1:
                reasoning_end = len(response)
            reasoning = response[reasoning_start:reasoning_end].strip()
        
        return verdict, reasoning, confidence


class MadrCorag(RagarCorag):
    """
    Multi-Agent Debate Refinement (MADR) claim verification system.
    
    Uses multiple LLM agents that independently analyze claims and debate
    through iterative refinement rounds to reach consensus.
    """
    
    def __init__(self, mc: ModelClient, num_agents: int = 3, max_debate_rounds: int = 3, 
                 consensus_threshold: float = 0.7):
        super().__init__(mc)

        self._searcher = LuceneSearcher(INDEX_DIR)
        
        # MADR-specific parameters
        self.num_agents = num_agents
        self.max_debate_rounds = max_debate_rounds
        self.consensus_threshold = consensus_threshold
        self.agents: List[MadrAgent] = [MadrAgent(i, mc) for i in range(num_agents)]
    
    def answer(self, question: str) -> str:
        """Retrieve evidence using BM25 search and answer using prompt template."""
        hits = self._searcher.search(question, k=5)  # Retrieve more evidence for MADR
        search_results = []
        for hit in hits:
            doc = self._searcher.doc(hit.docid)
            contents = doc.get("contents")
            if contents:
                search_results.append(contents)
        
        output = "\n\n".join(search_results)
        return self._mc.send_prompt("answer", [output, question]).strip()
    
    def stop_check(self, claim: str, qa_pairs: list[tuple[str, str]]) -> bool:
        """Check if enough evidence has been gathered using prompt template."""
        if len(qa_pairs) < 2:
            return False
        res = self._mc.send_prompt("stop_check", [claim, qa_pairs]).lower()
        return "conclusive" in res
    
    def verdict(self, claim: str, qa_pairs: list[tuple[str, str]]) -> tuple[int, str]:
        """
        Produce final verdict using multi-agent debate refinement.
        
        Returns:
            (verdict_int, detailed_output) where verdict_int is 0=REFUTES, 1=SUPPORTS, 2=NOT_ENOUGH_INFO
        """
        # Compile all evidence from QA pairs
        evidence = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])
        
        # Track debate history
        debate_history = []
        
        # Initial round - all agents independently analyze
        print(f"\n[MADR] Starting debate with {self.num_agents} agents...")
        agent_verdicts = []
        for agent in self.agents:
            verdict, reasoning, confidence = agent.analyze_claim(claim, evidence)
            agent_verdicts.append({
                'agent_id': agent.agent_id,
                'verdict': verdict,
                'reasoning': reasoning,
                'confidence': confidence
            })
        
        debate_history.append({
            'round': 0,
            'verdicts': agent_verdicts.copy()
        })
        
        # Debate rounds - agents refine based on others' positions
        for round_num in range(1, self.max_debate_rounds + 1):
            print(f"[MADR] Round {round_num}: Refining positions...")
            
            # Check for consensus
            if self._has_consensus(agent_verdicts):
                print(f"[MADR] Consensus reached in round {round_num}")
                break
            
            # Each agent refines their position
            refined_verdicts = []
            for agent in self.agents:
                verdict, reasoning, confidence = agent.analyze_claim(claim, evidence, agent_verdicts)
                refined_verdicts.append({
                    'agent_id': agent.agent_id,
                    'verdict': verdict,
                    'reasoning': reasoning,
                    'confidence': confidence
                })
            
            agent_verdicts = refined_verdicts
            debate_history.append({
                'round': round_num,
                'verdicts': agent_verdicts.copy()
            })
        
        # Aggregate final verdict using confidence-weighted voting
        final_verdict, final_reasoning = self._aggregate_verdicts(agent_verdicts, debate_history)
        
        return final_verdict, final_reasoning
    
    def _has_consensus(self, agent_verdicts: List[Dict]) -> bool:
        """Check if agents have reached consensus."""
        if not agent_verdicts:
            return False
        
        # Count verdicts
        verdict_counts = {0: 0, 1: 0, 2: 0}
        for av in agent_verdicts:
            verdict_counts[av['verdict']] += 1
        
        # Check if any verdict has sufficient majority
        total = len(agent_verdicts)
        for count in verdict_counts.values():
            if count / total >= self.consensus_threshold:
                return True
        
        return False
    
    def _aggregate_verdicts(self, agent_verdicts: List[Dict], debate_history: List[Dict]) -> Tuple[int, str]:
        """
        Aggregate agent verdicts using confidence-weighted voting.
        
        Returns:
            (final_verdict, explanation_json)
        """
        # Confidence-weighted voting
        verdict_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        for av in agent_verdicts:
            verdict_scores[av['verdict']] += av['confidence']
        
        # Select verdict with highest weighted score
        final_verdict = max(verdict_scores.items(), key=lambda x: x[1])[0]
        
        # Build detailed explanation
        verdict_labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
        
        explanation = {
            "final_verdict": verdict_labels[final_verdict],
            "confidence_weighted_scores": {
                verdict_labels[k]: round(v, 3) for k, v in verdict_scores.items()
            },
            "debate_rounds": len(debate_history),
            "agent_final_positions": [
                {
                    "agent_id": av['agent_id'],
                    "verdict": verdict_labels[av['verdict']],
                    "reasoning": av['reasoning'],
                    "confidence": round(av['confidence'], 3)
                }
                for av in agent_verdicts
            ],
            "consensus_reached": self._has_consensus(agent_verdicts)
        }
        
        return final_verdict, json.dumps(explanation, indent=2)
    
    def run(self, claim: str, max_iters: int = 3) -> dict:
        """
        Run MADR claim verification pipeline.
        
        Args:
            claim: The claim to verify
            max_iters: Maximum evidence gathering iterations
            
        Returns:
            Dictionary with claim, qa_pairs, verdict, and MADR-specific metadata
        """
        # Evidence gathering phase (same as baseline)
        qa_pairs = []
        question = self.init_question(claim)
        
        for i in range(max_iters):
            if i > 0:
                question = self.next_question(claim, qa_pairs)
            answer = self.answer(question)
            qa_pairs.append((question, answer))
            if self.stop_check(claim, qa_pairs):
                break
        
        # Multi-agent debate phase
        verdict, raw = self.verdict(claim, qa_pairs)
        
        # Parse debate details
        debate_details = json.loads(raw) if raw else {}
        
        return {
            "claim": claim,
            "qa_pairs": qa_pairs,
            "verdict": verdict,
            "verdict_raw": raw,
            "madr_metadata": debate_details
        }
