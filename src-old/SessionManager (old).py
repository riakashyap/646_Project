# Copyright (c) 2025 bdunahu, Ria. All rights reserved.
# Use of this source code is governed by an MIT license
# that can be found in the LICENSE file.

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from typing import Dict

class SessionManager():

  model_name: str
  max_to_generate: int
  top_k: int
  top_p: float
  temperature: float
  repetition_penalty: float

  _tokenizer: AutoTokenizer
  _model: AutoModelForCausalLM
  _config: AutoConfig

  _prompts: Dict[str, str] = {
    "initial_question": "You are an expert fact-checker given an unverified claim that needs to be explored.\n\nClaim: {}\nDate (your questions must be framed to be before this date: {}\n\nYou follow these Instructions:\n1: You understand the entire claim.\n\n2. You will make sure that the question is specific and focuses on one aspect of the claim (focus on one topic, should detail where, who, and what) and is very, very short.\n\n3. You should not appeal to video evidence nor ask for calculations or methodology.\n\n4. You are not allowed to use the word \"claim\". Instead, if you want to refer to the claim, you should point out the exact issue in the claim that you are phrasing your question around.\n\n5. You must never ask for calculations or methodology.\n\n6. Create a pointed factcheck question for the claim.\n\nReturn only the question.",
    "follow_up": "You are an expert fact-checker given an unverified claim and question-answer pairs regarding the claim that needs to be explored. You follow these steps:\n\nClaim: {}\nQuestion-Answer Pairs:\n{}\n\nAre you satisfied with the questions asked and do you have enough information to answer the claim?\n\nIf the answer to any of these questions is \"Yes\". then reply with only \"False\" or else answer, \"True\".",
    "secondary_question" : "You are given an unverified statement and question-answer pairs regarding the claim that needs to be explorted. You follow these steps:\n\nClaim: {}\nQuestion-Answer Pairs:\n{}\n\nYour task is to ask a followup question regarding the claim specifically based on the question answer pairs.\n\nNever ask for sources or publishing.\n\nThe follow-up question must be descriptive, specific to the claim, and very short, brief, and concise.\n\nThe follow-up question should not appeal to video evidence not ask for calculations or methodology.\n\nThe followup question should not be seeking to answer a previously asked question. It can however attempt to improve that question.\n\nYou are not allowed to use the word \"claim\" or \"statement\". Instead if you want to refer to the claim/statement, you should point out the exact issue in the claim/statement that you are phrasing your question around.\n\nReply only with the followup question and nothing else.",
    "adj_initial_question" : "You are an expert fact-checker given an unverified claim that needs to be explored. Your task is to create a factcheck question for the claim. To do so, you must 1) make sure the question is specific and focuses on one aspect of the claim (detail who, what, where, and when). Instead of using the word \"claim\", point out the exact issue in the claim you are phrasing your question around. Return only the question, and make sure it is very, very short.\n\nExample Claim: PPP on average provided a grant of around $11,000 per employee\nQ: How does the PPP define an \"employee\" for the purposes of calculating grants?\n\nClaim: {}\nQ: ",
    "adj_follow_up": "You are an expert fact-checker given an unverified claim and question-answer pairs regarding the claim that needs to be explored:\n\nClaim: {}\nQuestion-Answer Pairs:\n{}\n\nAre you satisfied with the questions asked and do you have enough information to answer the claim?\n\nIf the answer to both of these questions is \"Yes\". then reply with only \"True\" or else answer, \"False\". Do not give further reasoning.\n\nBinary answer: ",
    "adj+_follow_up": "You are an expert fact-checker given an unverified claim and question-answer pairs regarding the claim that needs to be explored:\n\nClaim: {}\nQuestion-Answer Pairs:\n{}\n\nAre you satisfied with the questions asked and do you have enough information to prove or disprove the claim?\n\nVery concisely explain how the question answer pairs succeed or fail to explain the claim, then conclude with \"True\" if you are satisfied with the questions asked, otherwise respond with \"False\"",
    "adj++_follow_up": "You are given an unverified claim and question-answer pairs regarding the claim that needs to be explored:\n\nClaim: {}\nQuestion-Answer Pairs:\n{}\n\nIn a single, concise, logical sentence, address if the question-evidence pairs fully confirm or disprove the claim.\n\nThen, conclude with \"Conclusive\" if you are satisfied with the questions asked and have enough information to answer the claim, otherwise respond with \"Inconclusive\". No other output is needed or warranted.",
  }

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

  def build_prompt(self, prompt_type: str, args: list[str]) -> str:
    """
    Given PROMPT_TYPE, which must be a key in SELF._PROMPTS, returns the respective prompt ready for LM inference.
    """
    prompt = self._prompts[prompt_type]
    return f"{prompt.format(*args)}\n\n"

  def parse_boolean_answer(self, response: str) -> bool | None:
    """
    Given RESPONSE, attempts to parse a binary value by searching the string for keywords 'True' or 'False'. Returns None if both or none are found.
    """
    lower = response.lower()
    if 'true' in lower and 'false' not in lower:
      return True
    elif 'false' in lower and 'true' not in lower:
      return False
    elif 'true' in lower and 'false' in lower:
      return None
    return None

  def step(self, context: str) -> str:
    """
    Given CONTEXT, prompts the loaded model and returns the response.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = self._tokenizer(context, return_tensors="pt").to(device)

    with torch.no_grad():
      outputs = self._model.generate(
        **inputs,
        max_new_tokens=self.max_to_generate,
        top_k=self.top_k,
        top_p=self.top_p,
        temperature=self.temperature,
        repetition_penalty=self.repetition_penalty
      )

    output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text[len(context):].strip()

