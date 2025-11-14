"""
Copyright:

  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  Unit tests for Multi-Agent Debate Refinement (MADR) implementation.

Code:
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.madr_corag import MadrCorag, MadrAgent
from src.model_clients import LlamaCppClient
from src.config import PROMPTS_DIR, INDEX_DIR


class TestMADRComponents(unittest.TestCase):
    """Test individual MADR components."""
    
    def setUp(self):
        """Set up test fixtures."""
        user_prompts_dir = PROMPTS_DIR / "madr"
        self.mc = LlamaCppClient(user_prompts_dir, None, think_mode_bool=False)
    
    def test_agent_creation(self):
        """Test that agents can be created properly."""
        agent = MadrAgent(0, self.mc)
        self.assertEqual(agent.agent_id, 0)
        self.assertIsNone(agent.current_verdict)
        self.assertEqual(agent.current_reasoning, "")
        self.assertEqual(agent.confidence, 0.0)
    
    def test_agent_response_parsing(self):
        """Test parsing of agent responses."""
        agent = MadrAgent(0, self.mc)
        
        # Test SUPPORTS verdict
        response = """
        VERDICT: SUPPORTS
        REASONING: The evidence clearly shows this is true.
        CONFIDENCE: 0.9
        """
        verdict, reasoning, confidence = agent._parse_agent_response(response)
        self.assertEqual(verdict, 1)
        self.assertIn("evidence", reasoning.lower())
        self.assertAlmostEqual(confidence, 0.9, places=1)
        
        # Test REFUTES verdict
        response = """
        VERDICT: REFUTES
        REASONING: The claim contradicts known facts.
        CONFIDENCE: 0.85
        """
        verdict, reasoning, confidence = agent._parse_agent_response(response)
        self.assertEqual(verdict, 0)
        
        # Test NOT_ENOUGH_INFO verdict
        response = """
        VERDICT: NOT_ENOUGH_INFO
        REASONING: Insufficient evidence to make determination.
        CONFIDENCE: 0.6
        """
        verdict, reasoning, confidence = agent._parse_agent_response(response)
        self.assertEqual(verdict, 2)
    
    def test_madr_initialization(self):
        """Test MADR system initialization."""
        madr = MadrCorag(self.mc, num_agents=3, max_debate_rounds=2)
        self.assertEqual(madr.num_agents, 3)
        self.assertEqual(madr.max_debate_rounds, 2)
        self.assertEqual(len(madr.agents), 3)
        self.assertIsNotNone(madr._searcher)
    
    def test_consensus_detection(self):
        """Test consensus detection logic."""
        madr = MadrCorag(self.mc, num_agents=3, consensus_threshold=0.7)
        
        # Test strong consensus (all agree)
        verdicts = [
            {'agent_id': 0, 'verdict': 1, 'reasoning': 'test', 'confidence': 0.9},
            {'agent_id': 1, 'verdict': 1, 'reasoning': 'test', 'confidence': 0.8},
            {'agent_id': 2, 'verdict': 1, 'reasoning': 'test', 'confidence': 0.85}
        ]
        self.assertTrue(madr._has_consensus(verdicts))
        
        # Test weak consensus (split decision)
        verdicts = [
            {'agent_id': 0, 'verdict': 1, 'reasoning': 'test', 'confidence': 0.9},
            {'agent_id': 1, 'verdict': 0, 'reasoning': 'test', 'confidence': 0.8},
            {'agent_id': 2, 'verdict': 2, 'reasoning': 'test', 'confidence': 0.85}
        ]
        self.assertFalse(madr._has_consensus(verdicts))
        
        # Test partial consensus (2/3 agree = 66.7%, below 70% threshold)
        verdicts = [
            {'agent_id': 0, 'verdict': 1, 'reasoning': 'test', 'confidence': 0.9},
            {'agent_id': 1, 'verdict': 1, 'reasoning': 'test', 'confidence': 0.8},
            {'agent_id': 2, 'verdict': 0, 'reasoning': 'test', 'confidence': 0.85}
        ]
        self.assertFalse(madr._has_consensus(verdicts))
    
    def test_verdict_aggregation(self):
        """Test confidence-weighted verdict aggregation."""
        madr = MadrCorag(self.mc, num_agents=3)
        
        # Test clear majority with confidence weighting
        verdicts = [
            {'agent_id': 0, 'verdict': 1, 'reasoning': 'Strong support', 'confidence': 0.9},
            {'agent_id': 1, 'verdict': 1, 'reasoning': 'Agrees', 'confidence': 0.85},
            {'agent_id': 2, 'verdict': 0, 'reasoning': 'Weak refutation', 'confidence': 0.4}
        ]
        
        final_verdict, explanation = madr._aggregate_verdicts(verdicts, [])
        self.assertEqual(final_verdict, 1)  # SUPPORTS should win
        self.assertIn('"final_verdict"', explanation)


class TestMADRIntegration(unittest.TestCase):
    """Integration tests for MADR with real claims (requires running LLM server)."""
    
    @classmethod
    def setUpClass(cls):
        """Set up for integration tests."""
        user_prompts_dir = PROMPTS_DIR / "madr"
        cls.mc = LlamaCppClient(user_prompts_dir, None, think_mode_bool=False)
    
    def test_simple_claim_verification(self):
        """Test MADR on a simple verifiable claim."""
        # Skip if index doesn't exist
        if not INDEX_DIR.exists():
            self.skipTest("Index not built")
        
        madr = MadrCorag(self.mc, num_agents=2, max_debate_rounds=2)
        
        # Use a simple claim
        claim = "Water freezes at 0 degrees Celsius."
        
        try:
            result = madr.run(claim, max_iters=2)
            
            # Check result structure
            self.assertIn('claim', result)
            self.assertIn('verdict', result)
            self.assertIn('qa_pairs', result)
            self.assertIn('madr_metadata', result)
            
            # Check verdict is valid
            self.assertIn(result['verdict'], [0, 1, 2])
            
            # Check metadata structure
            metadata = result['madr_metadata']
            self.assertIn('final_verdict', metadata)
            self.assertIn('agent_final_positions', metadata)
            self.assertIn('debate_rounds', metadata)
            
            print(f"\nTest claim: {claim}")
            print(f"MADR verdict: {['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO'][result['verdict']]}")
            print(f"Debate rounds: {metadata['debate_rounds']}")
            print(f"Consensus: {metadata.get('consensus_reached', 'N/A')}")
            
        except Exception as e:
            # If LLM server not running, skip gracefully
            if "Connection" in str(e) or "refused" in str(e):
                self.skipTest(f"LLM server not available: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main(verbosity=2)
