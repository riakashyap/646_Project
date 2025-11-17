"""
Copyright:

  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file tests the base functionality of the RagarCorag pipeline.

Code:
"""


from src.ragar_corag import RagarCorag
from src.config import PROMPTS_DIR
from tests.mock_model_client import MockModelClient
from typing import Tuple
import unittest

class TestCorag(unittest.TestCase):

    ragar: RagarCorag = None
    ragar_client: MockModelClient = None

    claim: str = "foxes are felines"
    question: str = "what is a feline?"
    qa_pairs: str = [("what is a feline?", "Felidae is the family of mammals in the"
                      " order Carnivora colloquially referred to as cats.")]

    @classmethod
    def setUpClass(self):
        super().setUpClass()

        ragar_user_prompts_dir = PROMPTS_DIR / "ragar"
        ragar_sys_prompts_dir = None

        self.ragar_client = \
            MockModelClient(ragar_user_prompts_dir, ragar_sys_prompts_dir)
        self.ragar = RagarCorag(self.ragar_client)

    @classmethod
    def get_expected_prompts(self, client: MockModelClient,
                             key, args: list[str]) -> Tuple[str, str]:
        user_prompt, system_prompt = client._prompts[key]
        return user_prompt.format(*args), system_prompt

    def test_ragar_inital_question(self):
        key = "init_question"
        self.ragar.init_question(self.claim)

        expected_user, _ = \
            self.get_expected_prompts(self.ragar_client, key, [self.claim])
        actual_user, actual_system = self.ragar_client.get_prompts()

        self.assertEqual(expected_user, actual_user)
        self.assertIsNone(actual_system)

    def test_ragar_answer(self):
        key = "answer"

        self.ragar.answer(self.question)

        expected_user, _ = \
            self.get_expected_prompts(self.ragar_client, key, ["", self.question])
        actual_user, actual_system = self.ragar_client.get_prompts()

        expected_split = expected_user.splitlines()
        for line in expected_split:
            self.assertIn(line, actual_user, "Question format missing from prompt!")

    def test_ragar_next_question(self):
        key = "next_question"

        self.ragar.next_question(self.claim, self.qa_pairs)

        expected_user, _ = \
            self.get_expected_prompts(self.ragar_client, key, [self.claim, self.qa_pairs])
        actual_user, actual_system = self.ragar_client.get_prompts()

        self.assertEqual(expected_user, actual_user)

    def test_ragar_stop_check_template(self):
        key = "stop_check"

        self.ragar.stop_check(self.claim, self.qa_pairs)

        expected_user, _ = \
            self.get_expected_prompts(self.ragar_client, key, [self.claim, self.qa_pairs])
        actual_user, actual_system = self.ragar_client.get_prompts()

        self.assertEqual(expected_user, actual_user)

    def test_ragar_stop_check_conclusive(self):
        self.ragar_client.set_ret("...conclusive...")
        self.assertTrue(self.ragar.stop_check(self.claim, self.qa_pairs))

    def test_ragar_stop_check_inconclusive(self):
        self.ragar_client.set_ret("...inconclusive...")
        self.assertFalse(self.ragar.stop_check(self.claim, self.qa_pairs))

    def test_ragar_stop_check_neither(self):
        self.ragar_client.set_ret("...neither...")
        self.assertFalse(self.ragar.stop_check(self.claim, self.qa_pairs))

    def test_ragar_stop_check_both(self):
        self.ragar_client.set_ret("...conclusive...inconclusive...")
        self.assertFalse(self.ragar.stop_check(self.claim, self.qa_pairs))

    def test_ragar_verdict_template(self):
        key = "verdict"
        self.ragar.verdict(self.claim, self.qa_pairs)

        expected_user, _ = \
            self.get_expected_prompts(self.ragar_client, key, [self.claim, self.qa_pairs])
        actual_user, actual_system = self.ragar_client.get_prompts()

        self.assertEqual(expected_user, actual_user)

    def test_ragar_verdict_false(self):
        expected_res = "...false..."

        self.ragar_client.set_ret(expected_res)
        verdict, res = self.ragar.verdict(self.claim, self.qa_pairs)

        self.assertEqual(0, verdict)
        self.assertEqual(expected_res, res)

    def test_ragar_verdict_true(self):
        expected_res = "...true..."

        self.ragar_client.set_ret(expected_res)
        verdict, res = self.ragar.verdict(self.claim, self.qa_pairs)

        self.assertEqual(1, verdict)
        self.assertEqual(expected_res, res)

    def test_ragar_verdict_nei(self):
        expected_res = "...inconclusive..."

        self.ragar_client.set_ret(expected_res)
        verdict, res = self.ragar.verdict(self.claim, self.qa_pairs)

        self.assertEqual(2, verdict)
        self.assertEqual(expected_res, res)

    def test_ragar_verdict_KO(self):
        expected_res = "...neither..."

        self.ragar_client.set_ret(expected_res)
        verdict, res = self.ragar.verdict(self.claim, self.qa_pairs)

        self.assertIsNone(verdict)
        self.assertEqual(expected_res, res)

    def test_ragar_run(self):
        response = "abc"

        self.ragar_client.set_ret(response)
        results = self.ragar.run(self.claim, 3)

        self.assertIsNone(results["verdict"])
        self.assertEqual(response, results["verdict_raw"])
        self.assertEqual(self.claim, results["claim"])
        self.assertEqual(3, len(results["qa_pairs"]))
        self.assertEqual((response, response), results["qa_pairs"][0])
