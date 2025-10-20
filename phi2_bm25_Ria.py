# -*- coding: utf-8 -*-
"""
Fact-checking system using Microsoft Phi-2 model with BM25 retrieval

This script implements a fact-checking system that uses the Phi-2 model
to verify claims from the FEVER dataset. The system follows the RAGAR methodology:

1. Retrieve top-k relevant documents with BM25
2. Generate initial questions about the claim (model grounded on retrieved docs)
3. Answer questions iteratively
4. Check if enough information is gathered
5. Generate follow-up questions if needed
6. Provide final verification with rating (supported/refuted/failed)
"""

import subprocess
import sys
import os
import json
import time
import textwrap
from typing import Dict, List, Optional, Tuple
from collections import Counter

# Optional: install dependencies automatically (uncomment to use)
def install_packages():
    packages = [
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "tqdm",
        "numpy",
        "rank-bm25",
        "nltk"
    ]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CONFIG = {
    "model_name": "microsoft/phi-2",  # Phi-2 model identifier
    "num_samples": 5,
    "max_iters": 2,
    "max_to_generate": 120,
    "top_k": 40,
    "top_p": 0.85,
    "temperature": 0.6,
    "repetition_penalty": 1.05,
    # BM25 config
    "bm25_folder": "D:\CS646\mistral\wiki-pages\wiki-pages",
    "bm25_top_k": 5
}

# ---------------------------------------------------------------------
# Imports that require installed packages
# ---------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Ensure nltk punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ---------------------------------------------------------------------
# Language Model Client
# ---------------------------------------------------------------------
class TransformersLMClient():
    """
    Uses a Hugging Face AutoModelForCausalLM model for text generation.
    """

    def __init__(
        self,
        model_name: str,
        max_to_generate: int = 100,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1
    ) -> None:
        self.model_name = model_name
        self.max_to_generate = max_to_generate
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

        print(f"Loading model: {model_name} ... (this may take a while)")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with Phi-2 compatibility
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._config = AutoConfig.from_pretrained(model_name)
        self.device = device

    def send_query(self, context: str) -> str:
        """Prompt the model and return the generated text."""
        inputs = self._tokenizer(context, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_to_generate,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id
            )

        output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt prefix if model echoes it
        if output_text.startswith(context):
            output_text = output_text[len(context):].strip()
        return output_text.strip()


# ---------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------
class Prompt():
    _prompts: Dict[str, str] = {
        "initial_question": textwrap.dedent("""
            You are an expert fact-checker given an unverified claim that needs to be explored.

            Claim: {}

            Instructions:
            1. Understand the claim fully.
            2. Ask a single, specific, short question that investigates one factual aspect of the claim.
            3. Do not ask for videos, methods, or sources.
            Return only the question.
        """),

        "follow_up_check": textwrap.dedent("""
            Claim: {}
            Question-Answer Pairs:
            {}

            Do you have enough information to verify the claim?
            Reply with only "True" if more information is needed or "False" if enough is gathered.
        """),

        "follow_up_question": textwrap.dedent("""
            Claim: {}
            Question-Answer Pairs:
            {}

            Ask one short follow-up question that clarifies or adds missing context.
            Reply only with the follow-up question.
        """),

        "verify": textwrap.dedent("""
            You are a fact-checking assistant. Based on the following claim and question-answer pairs,
            decide whether the claim is supported, refuted, or failed (not enough information).

            Claim: {}
            Question-Answer Pairs:
            {}

            Output a JSON object with:
            - claim
            - rating ("supported", "refuted", or "failed")
            - factcheck (short explanation)
        """)
    }

    @staticmethod
    def build_prompt(prompt_type: str, args: List[str]) -> str:
        return Prompt._prompts[prompt_type].format(*args)


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def parse_boolean_answer(response: str) -> Optional[bool]:
    if response is None:
        return None
    lower = response.lower()
    if "true" in lower and "false" in lower:
        return None
    if "true" in lower:
        return True
    if "false" in lower:
        return False
    return None


def parse_rating_answer(answer: str) -> Optional[str]:
    if answer is None:
        return None
    lower = answer.lower()
    if '"supported"' in lower or 'supported' in lower:
        return "supported"
    if '"refuted"' in lower or 'refuted' in lower:
        return "refuted"
    if '"failed"' in lower or 'failed' in lower:
        return "failed"
    return None


# ---------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, folder_path: str, top_k: int = 5):
        """
        folder_path: directory with text/JSONL files containing Wikipedia or evidence documents
        top_k: how many documents to retrieve per query
        """
        self.top_k = top_k
        self.documents: List[str] = []
        self.titles: List[str] = []

        if not os.path.exists(folder_path):
            print(f"[BM25Retriever] Warning: folder '{folder_path}' does not exist. Retriever will be empty.")
            self.bm25 = None
            return

        # Load corpus: accept .jsonl, .json, .txt
        for fname in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, fname)
            if not os.path.isfile(path):
                continue
            if fname.endswith(".jsonl") or fname.endswith(".json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                # support common fields
                                text = data.get("text") or data.get("content") or data.get("body") or ""
                            except json.JSONDecodeError:
                                # fallback: treat line as plain text
                                text = line
                            if text:
                                self.documents.append(text)
                                self.titles.append(fname)
                except Exception as e:
                    print(f"[BM25Retriever] Error reading {path}: {e}")
            elif fname.endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            self.documents.append(text)
                            self.titles.append(fname)
                except Exception as e:
                    print(f"[BM25Retriever] Error reading {path}: {e}")
            else:
                # skip other file types
                continue

        if not self.documents:
            print(f"[BM25Retriever] No documents loaded from {folder_path}. Retriever will be empty.")
            self.bm25 = None
            return

        # Tokenize corpus (simple word_tokenize)
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[BM25Retriever] Loaded {len(self.documents)} documents. BM25 ready.")

    def retrieve(self, query: str) -> List[str]:
        """Return top-k most relevant texts for the given query"""
        if not self.bm25:
            return []
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        return [self.documents[i] for i in top_indices]


# ---------------------------------------------------------------------
# Verification Logic (integrated with BM25 retrieval)
# ---------------------------------------------------------------------
def verify_claim(client: TransformersLMClient, claim: str, retriever: Optional[BM25Retriever] = None, max_iters: int = 3) -> Optional[str]:
    # Step 1: Retrieve supporting evidence using BM25 (if available)
    context = ""
    retrieved_docs: List[str] = []
    if retriever is not None:
        retrieved_docs = retriever.retrieve(claim)
        if retrieved_docs:
            # join docs with separators and truncate each doc to a reasonable length for prompt
            safe_docs = []
            for doc in retrieved_docs:
                # keep only first ~800 chars of each retrieved doc to avoid overly long prompt
                safe_docs.append(doc.strip().replace("\n", " ")[:800])
            context = "\n\n---\n\n".join(safe_docs)

    if context:
        print(f"\n[verify_claim] Retrieved {len(retrieved_docs)} docs for claim (showing prefix):")
        for i, d in enumerate(retrieved_docs, 1):
            print(f"[{i}] {d[:200]}...\n")

    # Build the initial question prompt (include context if available)
    initial_prompt = Prompt.build_prompt("initial_question", [claim])
    if context:
        initial_prompt = f"Context:\n{context}\n\n" + initial_prompt

    # Ask initial question
    question = client.send_query(initial_prompt).strip()
    qa_pairs: List[Tuple[str, str]] = []

    # Iterative QA loop
    for _ in range(max_iters):
        answer_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = client.send_query(answer_prompt).strip()
        qa_pairs.append((question, answer))

        # Check if we have enough information
        check_prompt = Prompt.build_prompt("follow_up_check", [claim, qa_pairs])
        # include context for the check too
        if context:
            check_prompt = f"Context:\n{context}\n\n" + check_prompt

        check_response = client.send_query(check_prompt)
        done = parse_boolean_answer(check_response)
        # If parse_boolean_answer returns False => "False" means enough info gathered (stop)
        # If it returns True => more info needed (continue)
        # If None => ambiguous; break to avoid endless loop
        if done is False:
            break
        if done is None:
            # ambiguous signal — produce at most one follow-up question and then stop
            follow_up_prompt = Prompt.build_prompt("follow_up_question", [claim, qa_pairs])
            if context:
                follow_up_prompt = f"Context:\n{context}\n\n" + follow_up_prompt
            question = client.send_query(follow_up_prompt).strip()
            # Do one final iteration
            answer_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            answer = client.send_query(answer_prompt).strip()
            qa_pairs.append((question, answer))
            break

        # If done is True (more info needed), ask for a follow-up question
        follow_up_prompt = Prompt.build_prompt("follow_up_question", [claim, qa_pairs])
        if context:
            follow_up_prompt = f"Context:\n{context}\n\n" + follow_up_prompt
        question = client.send_query(follow_up_prompt).strip()

    # Final verification prompt (include context)
    verify_prompt = Prompt.build_prompt("verify", [claim, qa_pairs])
    if context:
        verify_prompt = f"Context:\n{context}\n\n" + verify_prompt

    verify_answer = client.send_query(verify_prompt)
    rating = parse_rating_answer(verify_answer)
    # For debugging, print the QA pairs and final raw verify_answer
    print(f"\n[verify_claim] QA pairs collected:")
    for q, a in qa_pairs:
        print(f"Q: {q}\nA: {a}\n")
    print(f"[verify_claim] Raw verify answer: {verify_answer}\nParsed rating: {rating}")
    return rating


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
def main():
    print(f"Loading Microsoft Phi-2 model: {CONFIG['model_name']}")
    TC = TransformersLMClient(
        model_name=CONFIG['model_name'],
        max_to_generate=CONFIG['max_to_generate'],
        top_k=CONFIG['top_k'],
        top_p=CONFIG['top_p'],
        temperature=CONFIG['temperature'],
        repetition_penalty=CONFIG['repetition_penalty']
    )

    # Initialize BM25 retriever (if folder exists)
    bm25_folder = CONFIG.get("bm25_folder", "data/wiki_pages")
    retriever = None
    if os.path.exists(bm25_folder):
        retriever = BM25Retriever(folder_path=bm25_folder, top_k=CONFIG.get("bm25_top_k", 5))
    else:
        print(f"[main] BM25 folder '{bm25_folder}' not found. Running model-only (no retrieval).")

    print("\nUsing sample claims for testing...\n")
    sample_claims = [
        {"claim": "Barack Obama was born in Hawaii.", "label": "supports"},
        {"claim": "The Earth is flat.", "label": "refutes"},
        {"claim": "Python is a programming language.", "label": "supports"},
        {"claim": "The sun revolves around the Earth.", "label": "refutes"},
        {"claim": "Water boils at 100 degrees Celsius at sea level.", "label": "supports"}
    ]

    num_samples = min(CONFIG['num_samples'], len(sample_claims))
    max_iters = CONFIG['max_iters']

    correct = 0
    errors = 0
    times = []
    preds, labels = [], []

    for i, item in enumerate(sample_claims[:num_samples]):
        claim, label = item["claim"], item["label"].lower()
        print(f"\nProcessing claim {i+1}/{num_samples}: {claim}")

        try:
            start = time.time()
            rating = verify_claim(TC, claim, retriever=retriever, max_iters=max_iters)
            end = time.time()
            times.append(end - start)

            if rating == "supported":
                pred = "supports"
            elif rating == "refuted":
                pred = "refutes"
            elif rating == "failed":
                pred = "not enough info"
            else:
                pred = None
                errors += 1

            preds.append(pred)
            labels.append(label)
            if pred == label:
                correct += 1

            print(f"Rating: {rating}, Predicted: {pred}, Actual: {label}, Time: {end-start:.2f}s")

        except Exception as e:
            print(f"Error processing claim: {e}")
            errors += 1
            preds.append(None)
            labels.append(label)

    accuracy = correct / num_samples if num_samples > 0 else 0
    avg_time = sum(times) / len(times) if times else 0

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.3f} ({correct}/{num_samples})")
    print(f"Average time per claim: {avg_time:.2f}s")
    print(f"Errors encountered: {errors}")
    print("\nPrediction distribution:")
    print(Counter(preds))
    print("Label distribution:")
    print(Counter(labels))


if __name__ == "__main__":
    main()
