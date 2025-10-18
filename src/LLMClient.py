# Copyright (c) 2025 bdunahu, Ria. All rights reserved.
# Use of this source code is governed by an MIT license
# that can be found in the LICENSE file.

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from typing import Dict
import time

class TransformersLMClient():
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

  def __init__(self,
               model_name: str,
               max_to_generate: int = 100,
               top_k: int = 50,
               top_p: float = 1.0,
               temperature: float = 0.7,
               repetition_penalty: float = 1.1) -> None:
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

    start = time.time()
    with torch.no_grad():
      outputs = self._model.generate(
        **inputs,
        max_new_tokens=self.max_to_generate,
        top_k=self.top_k,
        top_p=self.top_p,
        temperature=self.temperature,
        repetition_penalty=self.repetition_penalty
      )
    print(f'request took {time.time() - start} seconds')

    output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text[len(context):].strip()

import requests

class LlamaCppClient:
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

    def __init__(self,
                 system_message: str = "You are a helpful assistant.",
                 should_think: bool = False,
                 host: str = "127.0.0.1",
                 port: int = 4568,
                 temperature: float = 0.7,):
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
            start = time.time()
            response = requests.post(self.api, json=payload)
            response.raise_for_status()
            result = response.json()
            print(f'request took {time.time() - start} seconds')
            return result["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            raise requests.RequestException(
                f"HTTP request to llama-cpp server failed: {e}"
            )
        except (KeyError, IndexError) as e:
            raise requests.KeyError(
                f"Unexpected response format: {e}"
            )
        

ctxt1 = SM.build_prompt("initial_question", ["The titular Lord of the Rings was never bested by a dog.", "2000"])
q1 = SM.step(ctxt1)
print(ctxt1 + q1)

class PromptAnalyzer():

  _prompts: Dict[str, str] = {
    "initial_question": (),
    "follow_up": ("You are given an unverified claim and question-answer pairs regarding the claim that needs to be explored.\n\nIn a single, concise, logical sentence, address if the question-evidence pairs fully confirm or disprove the claim.\n\nThen, conclude with \"Conclusive\" if you are satisfied with the questions asked and have enough information to answer the claim, otherwise respond with \"Inconclusive\". No other output is needed or warranted.", "Claim: {}\nQuestion-Answer Pairs:\n{}"),
    "secondary_question" : (),
  }

  @staticmethod
  def get_system_message(prompt_type :str) -> str:
    return PromptAnalyzer._prompts[prompt_type][0]

  @staticmethod
  def build_template(prompt_type: str, args: list[str]) -> str:
    """
    Given PROMPT_TYPE, which must be a key in SELF._PROMPTS, returns the completed template.
    """
    prompt = PromptAnalyzer._prompts[prompt_type][1]
    return f"{prompt.format(*args)}"

  @staticmethod
  def build_prompt(prompt_type: str, args: list[str]) -> str:
    """
    Given PROMPT_TYPE, which must be a key in SELF._PROMPTS, returns the respective prompt ready for LM inference.
    """
    prompt = PromptAnalyzer.get_system_message(prompt_type)
    template = PromptAnalyzer.build_template(prompt_type, args)
    return f"{prompt}\n\n{template}"

  @staticmethod
  def parse_conclusivity(response: str) -> bool | None:
    """
    Given RESPONSE, attempts to parse a binary value by searching the string for keywords 'Conclusive' or 'Inconclusive'. Returns None if both or none are found.
    """
    lower = response.lower()
    
    has_conclusive = 'conclusive' in lower and 'inconclusive' not in lower
    has_inconclusive = 'inconclusive' in lower
    
    if has_conclusive and not has_inconclusive:
        return True
    elif has_inconclusive and not has_conclusive:
        return False
    return None

