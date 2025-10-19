import requests
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional
from pathlib import Path

class ModelClient(ABC):
    _prompts: Dict[str, str]

    def __init__(self):
        self._prompts = {}

    @abstractmethod
    def send_query(self, context: str) -> str:
        """Send a query to the model and return a response."""
        pass

    def register_prompt(self, key: str, rel_path: str):
        """Register a prompt file and optional parser."""
        base_dir = Path(__file__).parent
        full_path = base_dir / rel_path
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
            self._prompts[key] = content

    def send_prompt(self, key: str, args: list[str]):
        template = self._prompts[key]
        prompt = template.format(*args)
        return self.send_query(prompt)
        
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
    system_message: str

    def __init__(
        self,
        system_message: str = "You are a helpful assistant.",
        should_think: bool = False,
        host: str = "127.0.0.1",
        port: int = 4568,
        temperature: float = 0.7,
    ):
        super().__init__()
        self.api = f"http://{host}:{port}/v1/chat/completions"
        self.temperature = temperature
        self.should_think = should_think
        self.system_message = system_message

    def send_query(self, context: str) -> str:
        think = "" if self.should_think else "/no_think"
        messages = [
            {"role": "system", "content": self.system_message + think},
            {"role": "user", "content": context},
        ]

        payload = {
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

    def send_query(self, context: str) -> str:
        """
        Given CONTEXT, prompts the loaded model and returns the response.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self._tokenizer(context, return_tensors="pt").to(device)

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