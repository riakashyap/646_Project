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

from . import config
from .config import INDEX_DIR
from .ragar_corag import RagarCorag
from .model_clients import ModelClient
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

        if config.LOGGER:
            config.LOGGER.info(f"\n[MADR/TRACE] Agent {self.agent_id} analyzing claim...")
            config.LOGGER.info(f"[MADR/TRACE] Evidence length: {len(evidence)} chars")

        if other_verdicts is None or len(other_verdicts) == 0:
            # Initial analysis - use agent_initial_analysis prompt
            response = self._mc.send_prompt("agent_initial_analysis", [
                str(self.agent_id),
                claim,
                evidence
            ])
        else:
            if config.LOGGER:
                config.LOGGER.info(f"[MADR/TRACE] Agent {self.agent_id} refining based on other agents")

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

        if config.LOGGER:
            config.LOGGER.info(f"[MADR/TRACE] Agent {self.agent_id} Verdict: {verdict}  Confidence: {confidence:.2f}")

        self.current_verdict = verdict
        self.current_reasoning = reasoning
        self.confidence = confidence
        
        return verdict, reasoning, confidence
    
    def _parse_agent_response(self, response: str) -> Tuple[int, str, float]:
        """Parse agent response robustly without relying on strict formatting."""
        verdict = 2  # default: NOT_ENOUGH_INFO
        reasoning_lines = []
        confidence = 0.5  # default fallback

        lines = response.strip().split("\n")
        for line in lines:
            clean = line.strip()
            upper = clean.upper()

            # VERDICT
            if upper.startswith("VERDICT"):
                text = clean.split(":", 1)[-1].strip().upper()
                if "REFUTE" in text or "FALSE" in text:
                    verdict = 0
                elif "SUPPORT" in text or "TRUE" in text:
                    verdict = 1
                else:
                    verdict = 2

            # CONFIDENCE
            elif upper.startswith("CONFIDENCE"):
                try:
                    text = clean.split(":", 1)[-1].strip()
                    confidence = float(text)
                    confidence = max(0.0, min(1.0, confidence))  # clamp
                except ValueError:
                    confidence = 0.5

            # REASONING
            elif "REASON" in upper:
                reasoning_lines.append(clean)

             # fallback: treat any non-empty line as reasoning
            elif clean:
                reasoning_lines.append(clean)

        reasoning = "\n".join(reasoning_lines).strip()

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

        self._searcher = LuceneSearcher(str(INDEX_DIR))
        
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
    
    def stop_check(self, claim: str, qa_pairs: List[Tuple[str, str]]) -> bool:
        """
        Determine whether enough evidence has been gathered to stop asking questions.
        MADR stops early when the agents already reach consensus based on current evidence.
        """
        if len(qa_pairs) < 2:
            return False

        # Compile evidence so far
        evidence = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])

        # Ask all MADR agents for their current position
        agent_verdicts = []
        for agent in self.agents:
            verdict, reasoning, confidence = agent.analyze_claim(claim, evidence)
            agent_verdicts.append({
                "agent_id": agent.agent_id,
                "verdict": verdict,
                "reasoning": reasoning,
                "confidence": confidence
            })

        if config.LOGGER:
            config.LOGGER.info("[MADR/TRACE] stop_check: agent verdicts collected")
            for av in agent_verdicts:
                config.LOGGER.info(
                    f"[MADR/TRACE]  Agent {av['agent_id']} → {av['verdict']} "
                    f"(conf {av['confidence']:.2f})"
                )

        # Stop if MADR agents already reached consensus
        if self._has_consensus(agent_verdicts):
            if config.LOGGER:
                config.LOGGER.info("[MADR/TRACE] stop_check: consensus reached → stopping early")
            return True
        return False

    def verdict(self, claim: str, qa_pairs: list[tuple[str, str]]) -> tuple[int, str]:
        """
        Produce final verdict using multi-agent debate refinement.
        
        Returns:
            (verdict_int, detailed_output) where verdict_int is 0=REFUTES, 1=SUPPORTS, 2=NOT_ENOUGH_INFO
        """
        
        if config.LOGGER:
            config.LOGGER.info("[MADR/TRACE] Starting MADR Debate")
            config.LOGGER.info(f"[MADR/TRACE] Claim: {claim}")
            config.LOGGER.info(f"[MADR/TRACE] Evidence Q/A pairs: {len(qa_pairs)}")

        # Compile all evidence from QA pairs
        evidence = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])
        
        # Track debate history
        debate_history = []
        
        # Initial round - all agents independently analyze

        if config.LOGGER:
            config.LOGGER.info(f"\n[MADR] Starting debate with {self.num_agents} agents...")
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
            if config.LOGGER:
                config.LOGGER.info(f"[MADR] Round {round_num}: Refining positions...")

            # Check for consensus
            if self._has_consensus(agent_verdicts):
                if config.LOGGER:
                    config.LOGGER.info(f"[MADR] Consensus reached in round {round_num}")
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
                if config.LOGGER:
                    config.LOGGER.info(f"[MADR/TRACE] Agent {agent.agent_id} updated verdict → {verdict} (conf {confidence:.2f})")

            agent_verdicts = refined_verdicts
            debate_history.append({
                'round': round_num,
                'verdicts': agent_verdicts.copy()
            })
        
        # Aggregate final verdict using confidence-weighted voting
        if config.LOGGER:
            config.LOGGER.info("\n[MADR/TRACE] Debate complete. Aggregating final verdict...\n")
        final_verdict, final_reasoning = self._aggregate_verdicts(agent_verdicts, debate_history)

        if config.LOGGER:
            config.LOGGER.info(f"[MADR/TRACE] Final Verdict: {final_verdict}")
        return final_verdict, final_reasoning
    
    def _has_consensus(self, agent_verdicts: List[Dict]) -> bool:
        """Check if agents have reached consensus."""

        if config.LOGGER:
            config.LOGGER.info("\n[MADR/TRACE] Checking consensus among agents...")

        if not agent_verdicts:
            if config.LOGGER:
                config.LOGGER.info("[MADR/TRACE] No agent verdicts present — cannot form consensus.")
            return False
        
        # Count verdicts
        verdict_counts = {0: 0, 1: 0, 2: 0}
        for av in agent_verdicts:
            verdict_counts[av['verdict']] += 1
        
        total = len(agent_verdicts)
        if config.LOGGER:
            config.LOGGER.info(f"[MADR/TRACE] Verdict counts: "
                f"REFUTES={verdict_counts[0]}, SUPPORTS={verdict_counts[1]}, NEI={verdict_counts[2]}")
            config.LOGGER.info(f"[MADR/TRACE] Consensus threshold: {self.consensus_threshold:.2f}")

        # Check majority threshold
        for verdict, count in verdict_counts.items():
            ratio = count / total
            if config.LOGGER:
                config.LOGGER.info(f"[MADR/TRACE] Verdict {verdict} ratio: {ratio:.2f}")

            if ratio >= self.consensus_threshold:
                if config.LOGGER:
                    config.LOGGER.info(f"[MADR/TRACE] Consensus achieved on verdict {verdict} (ratio {ratio:.2f})")
                return True

        if config.LOGGER:
            config.LOGGER.info("[MADR/TRACE] No consensus yet.")
        return False
    
    def _aggregate_verdicts(self, agent_verdicts: List[Dict], debate_history: List[Dict]) -> Tuple[int, str]:
        """
        Aggregate agent verdicts using confidence-weighted voting.
        
        Returns:
            (final_verdict, explanation_json)
        """
        if config.LOGGER:
            config.LOGGER.info("\n[MADR/TRACE] Aggregating final verdict via confidence-weighted voting...")

        # Confidence-weighted voting
        verdict_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        for av in agent_verdicts:
            verdict_scores[av['verdict']] += av['confidence']

        if config.LOGGER:
            config.LOGGER.info("[MADR/TRACE] Weighted scores before selection:")
            config.LOGGER.info(f"  REFUTES: {verdict_scores[0]:.3f}")
            config.LOGGER.info(f"  SUPPORTS: {verdict_scores[1]:.3f}")
            config.LOGGER.info(f"  NEI:      {verdict_scores[2]:.3f}")
        # Select verdict with highest weighted score
        final_verdict = max(verdict_scores.items(), key=lambda x: x[1])[0]
        
        # Build detailed explanation
        verdict_labels = ["REFUTES", "SUPPORTS", "NOT_ENOUGH_INFO"]

        if config.LOGGER:
            config.LOGGER.info(f"[MADR/TRACE] Final aggregated verdict: {final_verdict} ({verdict_labels[final_verdict]})")

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
        if config.LOGGER:
            config.LOGGER.info("[MADR/TRACE] Aggregation complete.\n")
        
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

        # Call the parent class run() (RagarCorag.run)
        result = super().run(claim, max_iters)

        # Extract the raw MADR debate JSON (your verdict() produces this)
        raw = result.get("verdict_raw")

        # Parse metadata if present
        debate_details = json.loads(raw) if raw else {}

        # Add metadata to result
        result["madr_metadata"] = debate_details

        return result