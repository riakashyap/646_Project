# src/agentic_rag/phi2_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Phi2Agent:
    def __init__(self, model_name="microsoft/phi-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )

    def reason(self, claim, evidence_list):
        evidence_text = "\n\n".join(evidence_list[:3])
        prompt = f"""
You are a fact-checking assistant using evidence from Wikipedia.

Claim: {claim}

Evidence:
{evidence_text}

Based on the evidence, decide if the claim is:
- SUPPORTED
- REFUTED
- NOT ENOUGH INFO

Answer with one of the labels only.
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=64)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
