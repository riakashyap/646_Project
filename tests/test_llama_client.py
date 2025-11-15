"""
Copyright:

  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file tests the request interface (send query) of the llama-client.

Code:
"""

from src.model_clients import LlamaCppClient
import requests
import unittest

class MockResponse:
    def __init__(self, content):
        self.content = content

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": self.content
                    }
                }
            ]
        }

    def raise_for_status(self):
        pass

class TestLlamaClient(unittest.TestCase):

    expected_content: str
    user_prompts_dir: str
    system_prompts_dir: str
    port: int
    host: str
    expected_url: str
    expected_model: str
    expected_temperature: float
    expected_system_prompt: str
    expected_user_prompt: str
    think_mode: bool

    @classmethod
    def setUpClass(self):
        super().setUpClass()

        self.expected_model = "local"
        self.user_prompts_dir = None
        self.sys_prompts_dir = None

    def setUp(self):
        def requests_post_replacement(actual_url, data=None, json=None, **kwargs):
            self.assertEqual(self.expected_url, actual_url)

            self.assertEqual(self.expected_model, json["model"])

            # optional system prompt (defaults to)
            expected_system_prompt = "You are an expert fact-checker." \
                if self.expected_system_prompt is None else self.expected_system_prompt
            # think comment should have been appended to system_prompt
            expected_system_prompt = expected_system_prompt + "/no_think" \
                if not self.think_mode else expected_system_prompt
            messages = json["messages"]
            self.assertEqual(2, len(messages))
            self.assertEqual("system", messages[0]["role"])
            self.assertEqual("user", messages[1]["role"])
            self.assertEqual(expected_system_prompt, messages[0]["content"])
            self.assertEqual(self.expected_user_prompt, messages[1]["content"])

            self.assertEqual(self.expected_temperature, json["temperature"])

            return MockResponse(self.expected_content)

        # monkey patch requests.post
        requests.post = requests_post_replacement

        # default environment
        self.expected_content = "Success!"
        self.port = 4141
        self.host = "localhost"
        self.expected_url = f"http://{self.host}:{self.port}/v1/chat/completions"
        self.think_mode = False
        self.expected_temperature = 0.7
        self.expected_system_prompt = "You are doing well!"
        self.expected_user_prompt = "I am doing well!"

    def makeLlamaClient(self):
        return LlamaCppClient(
            self.user_prompts_dir,
            self.sys_prompts_dir,
            think_mode_bool = self.think_mode,
            host = self.host,
            port = self.port,
            temperature = self.expected_temperature,
        )

    def test_default_environment(self):
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(self.expected_content, actual_content)

    def test_variable_temperature(self):
        self.expected_temperature = 100.0
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(self.expected_content, actual_content)

    def test_think_on(self):
        # simulates thinking in the response. Should trim think tags.
        self.think_mode = True
        expected_content = "I finished thinking."
        self.expected_content = "<think>I am thinking.</think>" + expected_content
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(expected_content, actual_content)

    def test_variable_system(self):
        self.expected_system_prompt = "This is a system message."
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(self.expected_content, actual_content)

    def test_variable_user(self):
        self.expected_user_prompt = "This is a user message."
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(self.expected_content, actual_content)

    def test_variable_content(self):
        self.expected_content = "This is a response."
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(self.expected_content, actual_content)

    def test_default_system(self):
        self.expected_system_prompt = None
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(self.expected_content, actual_content)

    def test_default_system(self):
        self.expected_system_prompt = None
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(self.expected_content, actual_content)

    def test_default_system_think(self):
        self.expected_system_prompt = None
        self.think_mode = True
        lcpp = self.makeLlamaClient()
        actual_content = lcpp.send_query(
            self.expected_user_prompt,
            self.expected_system_prompt,
        )
        self.assertEqual(self.expected_content, actual_content)
