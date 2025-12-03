"""
Copyright:

    Copyright © 2025 Ria

    You should have received a copy of the MIT license along with this file.
    If not, see https://mit-license.org/

Commentary:

   This file tests the base functionality of the MADR pipeline.

Code:
"""

from src.madr import run_madr
from src.config import RAGAR_DIR, MADR_DIR
from tests.mock_model_client import MockModelClient, SeqMockClient
import unittest
from src.utils import parse_boolean, parse_ternary, parse_conclusive, get_prompt_files

class TestMadr(unittest.TestCase):

    claim = "Did Fritz Edmunds compile the 'redneck' list?"
    qa_pairs = [("q1", "a1"), ("q2", "a2")]
    explanation = "An initial explanation"

    @classmethod
    def get_expected_prompts(cls, client: MockModelClient, key, args: list[str]):
        user_prompt, system_prompt = client._prompts[key]
        return user_prompt.format(*args), system_prompt

    @classmethod
    def setUpClass(cls):
        cls.prompt_files = get_prompt_files(RAGAR_DIR, MADR_DIR)
        super().setUpClass()

    def test_run_madr_full_flow(self):
        # Response sequence: f1_init, f2_init, judge(False), f1_cross, f2_cross, judge(True), final_revise
        responses = [
            "f1_init",
            "f2_init",
            "False",
            "f1_cross",
            "f2_cross",
            "True",
            "final_revised",
        ]

        client = SeqMockClient(self.prompt_files, responses)

        final = run_madr(client, self.claim, self.qa_pairs, self.explanation, max_iters=2)

        self.assertEqual("final_revised", final)

        # Validate that initial prompts were formatted with the claim/qa_pairs/explanation
        exp1, _ = self.get_expected_prompts(client, "init_fb1", [self.claim, self.qa_pairs, self.explanation])
        exp2, _ = self.get_expected_prompts(client, "init_fb2", [self.claim, self.qa_pairs, self.explanation])

        sent_user_prompts = [h[0] for h in client.history]

        self.assertIn(exp1.splitlines()[0], sent_user_prompts[0])
        self.assertIn(exp2.splitlines()[0], sent_user_prompts[1])

        # Ensure judge was called and contained the initial f1/f2 text
        self.assertIn("f1_init", sent_user_prompts[2])

        # Final revise prompt should contain the original explanation
        self.assertIn(self.explanation, sent_user_prompts[-1])

    def test_run_madr_immediate_judge_true(self):
        # Sequence: f1_init, f2_init, judge(True), final_revise
        responses = ["f1_init", "f2_init", "True", "final_revised"]
        client = SeqMockClient(self.prompt_files, responses)

        final = run_madr(client, self.claim, self.qa_pairs, self.explanation, max_iters=3)

        self.assertEqual("final_revised", final)
        self.assertIn(self.explanation, client.history[-1][0])

    def test_prompt_templates_match_expected(self):
        # Build a predictable response queue so we can assert exact prompt formatting
        responses = ["f1_init", "f2_init", "False", "f1_cross", "f2_cross", "True", "final_revised"]
        client = SeqMockClient(self.prompt_files, responses.copy())

        _ = run_madr(client, self.claim, self.qa_pairs, self.explanation, max_iters=2)

        # init_fb1
        exp1, _ = self.get_expected_prompts(client, "init_fb1", [self.claim, self.qa_pairs, self.explanation])
        self.assertEqual(exp1, client.history[0][0])

        # init_fb2
        exp2, _ = self.get_expected_prompts(client, "init_fb2", [self.claim, self.qa_pairs, self.explanation])
        self.assertEqual(exp2, client.history[1][0])

        # judge should receive f1_init, f2_init
        f1 = responses[0]
        f2 = responses[1]
        exp_j1, _ = self.get_expected_prompts(client, "judge", [f1, f2])
        self.assertEqual(exp_j1, client.history[2][0])

        # cross feedbacks: first cross receives (f1_init, f2_init), second receives (f2_init, f1_init)
        exp_cross1, _ = self.get_expected_prompts(client, "cross_fb", [responses[0], responses[1]])

        # The second cross feedback call receives the revised feedback returned by the first cross step
        exp_cross2, _ = self.get_expected_prompts(client, "cross_fb", [responses[1], responses[3]])
        self.assertEqual(exp_cross1, client.history[3][0])
        self.assertEqual(exp_cross2, client.history[4][0])

        # final revise prompt contains f1_cross, f2_cross and explanation
        f1_final = responses[3]
        f2_final = responses[4]
        exp_rev, _ = self.get_expected_prompts(client, "revise", [f1_final, f2_final, self.explanation])
        self.assertEqual(exp_rev, client.history[-1][0])

    def test_run_madr_call_counts(self):
        # Ensure the number of model calls matches the MADR flow we expect
        responses = ["f1_init", "f2_init", "False", "f1_cross", "f2_cross", "True", "final_revised"]
        client = SeqMockClient(self.prompt_files, responses.copy())

        _ = run_madr(client, self.claim, self.qa_pairs, self.explanation, max_iters=2)

        # Expected calls: init_fb1, init_fb2, judge, cross_fb, cross_fb, judge
        self.assertEqual(7, len(client.history))


class TestParsers(unittest.TestCase):
    """Parser unit tests embedded into the MADR test file"""

    def test_parse_boolean_true(self):
        self.assertTrue(parse_boolean("True"))

    def test_parse_boolean_false(self):
        self.assertFalse(parse_boolean("False"))

    def test_parse_boolean_ambiguous(self):
        self.assertFalse(parse_boolean("This response contains True and False"))

    def test_parse_boolean_case_insensitive(self):
        self.assertTrue(parse_boolean("tRuE"))

    def test_parse_ternary_false(self):
        self.assertEqual(0, parse_ternary("False"))

    def test_parse_ternary_true(self):
        self.assertEqual(1, parse_ternary("True"))

    def test_parse_ternary_inconclusive(self):
        self.assertEqual(2, parse_ternary("Inconclusive"))

    def test_parse_ternary_precedence(self):
        # Matches current parsers.py behavior: false checked before true and inconclusive
        self.assertEqual(0, parse_ternary("This answer mentions False and Inconclusive"))

    def test_parse_ternary_none_when_missing(self):
        self.assertIsNone(parse_ternary("No verdict here"))

    def test_parse_conclusive_detects(self):
        self.assertTrue(parse_conclusive("This is conclusive"))

    def test_parse_conclusive_inconclusive_false(self):
        self.assertFalse(parse_conclusive("This is inconclusive"))


if __name__ == "__main__":
    unittest.main()
