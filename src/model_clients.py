"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric
  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file provides various model clients to interface with a local large
  language model.

Code:
"""

from abc import ABC, abstractmethod
from typing import Dict
import os
import re
import requests


class ModelClient(ABC):
    _prompts: Dict[str, str] = dict()

    def __init__(self, user_prompts_dir: str, system_prompts_dir: str | None = None):

        for file_name in os.listdir(user_prompts_dir):
            if file_name.endswith(".txt"):
                user_prompt_path = user_prompts_dir / file_name
                with open(user_prompt_path, "r", encoding="utf-8") as file:
                    user_prompt = file.read()

                system_prompt = None
                if system_prompts_dir is not None:
                    system_prompt_path = system_prompts_dir / file_name
                    with open(system_prompt_path, "r", encoding="utf-8") \
                         as file:
                        system_prompt = file.read()

                key = os.path.splitext(file_name)[0]
                self._prompts[key] = (user_prompt, system_prompt)

    @abstractmethod
    def send_query(self, user_prompt: str, system_prompt: str | None = None) -> str:
        """Send USER_PROMPT and SYSTEM_PROMPT if given to the model and return
        a response."""
        pass

    def send_prompt(self, key: str, args: list[str]):
        """Given KEY, retrieves the system message and constructs the user
        prompt, passing it to `send_query`"""

        # Let this throw an error if key is missing
        user_prompt, system_prompt = self._prompts[key]
        return self.send_query(user_prompt.format(*args), system_prompt)


class LlamaCppClient(ModelClient):
    """
    The LlamaCppClient utilizes the llama-cpp LLM-inference toolkit as the backend
    from Hugging Face's `transformers' library.

    The models compatible with llama-cpp are found here:
    https://huggingface.co/models?library=gguf&sort=trending

    Assumes LCPP server is running.
    E.g. llama-server --reasoning-budget 0 --port 4568 -t 8 -m /path/to/model.gguf
    """

    api: str
    think_mode: str
    temperature: float

    def __init__(
        self,
        user_prompts_dir: str,
        system_prompts_dir: str | None = None,
        think_mode_bool: bool = False,
        host: str = "127.0.0.1",
        port: int = 4568,
        temperature: float = 0.7,
     ):
        super().__init__(user_prompts_dir, system_prompts_dir)
        self.api = f"http://{host}:{port}/v1/chat/completions"
        self.temperature = temperature
        self.think_mode = "" if think_mode_bool else "/no_think"
        if think_mode_bool:
            print(f'{type(self).__name__} will think!')

    def send_query(self, user_prompt: str,
                   system_prompt: str | None = None) -> str:

        system_prompt = system_prompt if system_prompt is not None else \
            "You are an expert fact-checker."
        messages = [
            {"role": "system", "content": system_prompt + self.think_mode},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": "local",  # Ignored
            "messages": messages,
            "temperature": self.temperature
        }

        try:
            res = requests.post(self.api, json=payload)
            res.raise_for_status()
            result = res.json()
            content = result["choices"][0]["message"]["content"]
            return re.sub(r'<think>(?s:.)*</think>', '', content)

        except requests.RequestException as e:
            raise requests.RequestException(f"HTTP request to llama-cpp server failed: {e}")

        except (KeyError, IndexError) as e:
            raise requests.KeyError(f"Unexpected response format: {e}")