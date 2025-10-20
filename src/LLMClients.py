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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict
import os
import re
import requests
import torch


class ModelClient(ABC):
    _prompts: Dict[str, str] = dict()

    def __init__(self):
        """Initializes ModelClient by reading the available prompts."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        system_dir = os.path.join(base_dir, "assets", "prompts", "system_messages")
        template_dir = os.path.join(base_dir, "assets", "prompts", "templates")

        for file in os.listdir(system_dir):
            if file.endswith(".txt"):
                base_name = os.path.splitext(file)[0]

                system_path = os.path.join(system_dir, file)
                template_path = os.path.join(template_dir, file)

                if not os.path.exists(template_path):
                    raise FileNotFoundError(f"Template file not found for: {file}")

                with open(system_path, "r", encoding="utf-8") as sys_file:
                    system_message = sys_file.read()

                with open(template_path, "r", encoding="utf-8") as tmpl_file:
                    template_message = tmpl_file.read()

                self._prompts[base_name] = (system_message, template_message)

    @abstractmethod
    def send_query(self, system_message: str, user_message: str) -> str:
        """Send SYSTEM_MESSAGE and USER_MESSAGE to the model and return a response."""
        pass

    def send_prompt(self, key: str, args: list[str]):
        """Given KEY, retrieves the system message and constructs the user message,
        passing it to `send_query'"""
        # let this throw an error if key is missing
        system_message = self._prompts[key][0]
        user_message = self._prompts[key][1].format(*args)
        return self.send_query(system_message, user_message)


class LlamaCppClient(ModelClient):
    """
    The LlamaCppClient utilizes the llama-cpp LLM-inference toolkit as the backend
    from Hugging Face's `transformers' library.

    The models compatible with llama-cpp are found here:
    https://huggingface.co/models?library=gguf&sort=trending
    """

    api: str
    should_think: bool
    temperature: float

    def __init__(
        self,
        should_think: bool = False,
        host: str = "127.0.0.1",
        port: int = 4568,
        temperature: float = 0.7,
    ):
        super().__init__()
        self.api = f"http://{host}:{port}/v1/chat/completions"
        self.temperature = temperature
        self.should_think = should_think

    def send_query(self, system_message: str, user_message: str) -> str:
        think = "" if self.should_think else "/no_think"
        messages = [
            {"role": "system", "content": system_message + think},
            {"role": "user", "content": user_message},
        ]

        payload = {
            # model is ignored
            "model": "local",
            "messages": messages,
            "temperature": self.temperature
        }

        try:
            response = requests.post(self.api, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return re.sub(r'<think>\s*</think>', '', content)
        except requests.RequestException as e:
            raise requests.RequestException(f"HTTP request to llama-cpp server failed: {e}")
        except (KeyError, IndexError) as e:
            raise requests.KeyError(f"Unexpected response format: {e}")


class TransformersLMClient(ModelClient):
    """
    The TransformersLMClient utilizes the AutoModelForCausalLM class
    from Hugging Face's `transformers' library.

    The constructor takes the name of the model, which should come from
    this list:
    https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:3B,max:6B&sort=likes
    """

    model_name: str
    max_to_generate: int
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float

    _tokenizer: AutoTokenizer
    _model: AutoModelForCausalLM
    _config: AutoConfig

    def __init__(
        self,
        model_name: str,
        max_to_generate: int = 100,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_to_generate = max_to_generate
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self._config = AutoConfig.from_pretrained(model_name)

    def send_query(self, system_message: str, user_message: str) -> str:
        """
        Given CONTEXT, prompts the loaded model and returns the response.
        """
        # transformers library does not accept a system message
        payload = f'{system_message}\n\n{user_message}'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self._tokenizer(payload, return_tensors="pt").to(device)

        # start = time.time()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_to_generate,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty
            )
        # print(f'request took {time.time() - start} seconds')

        output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text[len(context):].strip()
