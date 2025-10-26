# -*- coding: utf-8 -*-
"""
Fact-checking system using Microsoft Phi-2 model with LuceneSearcher (cached index)

This script:
1. Builds a Lucene index from a Wikipedia/text folder ONCE and reuses it thereafter.
2. Uses Microsoft Phi-2 model for reasoning.
3. Implements RAG-style iterative fact verification.
"""

import os
import json
import time
import textwrap
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from collections import Counter

# ---------------------------------------------------------------------
# Optional: Auto-install dependencies
# ---------------------------------------------------------------------
def install_packages():
    packages = [
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "tqdm",
        "numpy",
        "pyserini",
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
    "model_name": "microsoft/phi-2",
    "num_samples": 5,
    "max_iters": 2,
    "max_to_generate": 120,
    "top_k": 40,
    "top_p": 0.85,
    "temperature": 0.6,
    "repetition_penalty": 1.05,
    # Lucene config
    "bm25_folder": "D:\\CS646\\mistral\\wiki-pages\\wiki-pages",  # source folder
    "lucene_index_dir": "D:\\CS646\\mistral\\wiki-index",          # persistent Lucene index
    "bm25_top_k": 5
}

# ---------------------------------------------------------------------
# Imports that require installed packages
# ---------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from pyserini.search.lucene import LuceneSearcher
import nltk
nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------
# Language Model Client
# ---------------------------------------------------------------------
class TransformersLMClient:
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

        print(f"Loading model: {model_name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self.device = device

    def send_query(self, context: str) -> str:
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
        if output_text.startswith(context):
            output_text = output_text[len(context):].strip()
        return output_text.strip()


# ---------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------
class Prompt:
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
    if not response:
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
    if not answer:
        return None
    lower = answer.lower()
    if "supported" in lower:
        return "supported"
    if "refuted" in lower:
        return "refuted"
    if "failed" in lower or "not enough" in lower:
        return "failed"
    return None


# ---------------------------------------------------------------------
# Lucene Retriever (build once, then cached)
# ---------------------------------------------------------------------
class LuceneRetriever:
    def __init__(self, data_folder: str, index_dir: str, top_k: int = 5):
        """
        Builds the Lucene index once if not present and reuses it afterward.
        """
        from pyserini.index.lucene import LuceneIndexer

        self.top_k = top_k
        self.index_dir = index_dir

        # Build index if missing
        if not os.path.exists(index_dir):
            print(f"[LuceneRetriever] Building new Lucene index at: {index_dir}")
            os.makedirs(index_dir, exist_ok=True)

            # Prepare a minimal JSON collection for indexing
            collection_dir = os.path.join(index_dir, "json_collection")
            os.makedirs(collection_dir, exist_ok=True)

            count = 0
            for fname in os.listdir(data_folder):
                path = os.path.join(data_folder, fname)
                if not os.path.isfile(path):
                    continue
                if fname.endswith((".jsonl", ".json")):
                    with open(path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            try:
                                data = json.loads(line)
                                text = data.get("text") or data.get("content") or data.get("body") or ""
                                if text.strip():
                                    count += 1
                                    with open(os.path.join(collection_dir, f"doc_{count}.json"), "w", encoding="utf-8") as out:
                                        json.dump({"id": f"{fname}_{i}", "contents": text.strip()}, out)
                            except json.JSONDecodeError:
                                continue
                elif fname.endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            count += 1
                            with open(os.path.join(collection_dir, f"doc_{count}.json"), "w", encoding="utf-8") as out:
                                json.dump({"id": fname, "contents": text}, out)
            print(f"[LuceneRetriever] Prepared {count} JSON docs for indexing.")

            # Build Lucene index using Pyserini
            os.system(
                f'python -m pyserini.index.lucene '
                f'--collection JsonCollection '
                f'--input "{collection_dir}" '
                f'--index "{index_dir}" '
                f'--generator DefaultLuceneDocumentGenerator '
                f'--threads 4 --storeRaw'
            )
            print(f"[LuceneRetriever] Index built and cached at: {index_dir}")
        else:
            print(f"[LuceneRetriever] Using existing Lucene index: {index_dir}")

        # Load the searcher
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(k1=0.9, b=0.4)

    def retrieve(self, query: str) -> List[str]:
        hits = self.searcher.search(query, k=self.top_k)
        return [hit.raw for hit in hits if hit.raw]


# ---------------------------------------------------------------------
# Verification Logic
# ---------------------------------------------------------------------
def verify_claim(client: TransformersLMClient, claim: str, retriever: Optional[LuceneRetriever] = None, max_iters: int = 3) -> Optional[str]:
    context = ""
    retrieved_docs: List[str] = []
    if retriever is not None:
        retrieved_docs = retriever.retrieve(claim)
        if retrieved_docs:
            safe_docs = [doc.strip().replace("\n", " ")[:800] for doc in retrieved_docs]
            context = "\n\n---\n\n".join(safe_docs)

    if context:
        print(f"\n[verify_claim] Retrieved {len(retrieved_docs)} docs for claim (showing prefix):")
        for i, d in enumerate(retrieved_docs, 1):
            print(f"[{i}] {d[:200]}...\n")

    initial_prompt = Prompt.build_prompt("initial_question", [claim])
    if context:
        initial_prompt = f"Context:\n{context}\n\n" + initial_prompt

    question = client.send_query(initial_prompt).strip()
    qa_pairs: List[Tuple[str, str]] = []

    for _ in range(max_iters):
        answer_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = client.send_query(answer_prompt).strip()
        qa_pairs.append((question, answer))

        check_prompt = Prompt.build_prompt("follow_up_check", [claim, qa_pairs])
        if context:
            check_prompt = f"Context:\n{context}\n\n" + check_prompt

        check_response = client.send_query(check_prompt)
        done = parse_boolean_answer(check_response)
        if done is False:
            break
        if done is None:
            follow_up_prompt = Prompt.build_prompt("follow_up_question", [claim, qa_pairs])
            if context:
                follow_up_prompt = f"Context:\n{context}\n\n" + follow_up_prompt
            question = client.send_query(follow_up_prompt).strip()
            answer_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            answer = client.send_query(answer_prompt).strip()
            qa_pairs.append((question, answer))
            break

        follow_up_prompt = Prompt.build_prompt("follow_up_question", [claim, qa_pairs])
        if context:
            follow_up_prompt = f"Context:\n{context}\n\n" + follow_up_prompt
        question = client.send_query(follow_up_prompt).strip()

    verify_prompt = Prompt.build_prompt("verify", [claim, qa_pairs])
    if context:
        verify_prompt = f"Context:\n{context}\n\n" + verify_prompt

    verify_answer = client.send_query(verify_prompt)
    rating = parse_rating_answer(verify_answer)
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

    retriever = LuceneRetriever(
        data_folder=CONFIG['bm25_folder'],
        index_dir=CONFIG['lucene_index_dir'],
        top_k=CONFIG['bm25_top_k']
    )

    sample_claims = [
        {"claim": "Barack Obama was born in Hawaii.", "label": "supports"},
        {"claim": "The Earth is flat.", "label": "refutes"},
        {"claim": "Python is a programming language.", "label": "supports"},
        {"claim": "The sun revolves around the Earth.", "label": "refutes"},
        {"claim": "Water boils at 100 degrees Celsius at sea level.", "label": "supports"}
    ]

    num_samples = min(CONFIG['num_samples'], len(sample_claims))
    correct = 0
    errors = 0
    times = []
    preds, labels = [], []

    for i, item in enumerate(sample_claims[:num_samples]):
        claim, label = item["claim"], item["label"].lower()
        print(f"\nProcessing claim {i+1}/{num_samples}: {claim}")
        try:
            start = time.time()
            rating = verify_claim(TC, claim, retriever, max_iters=CONFIG["max_iters"])
            end = time.time()
            if rating:
                preds.append(rating)
                labels.append(label)
                correct += int(rating == label)
            else:
                preds.append("failed")
                labels.append(label)
                errors += 1

            times.append(end - start)
            print(f"[Result] Claim: {claim}\nPredicted: {rating}\nTrue: {label}\nTime: {end - start:.2f}s\n")

        except Exception as e:
            errors += 1
            print(f"[Error] Failed processing claim: {claim}\n{e}")

    # Summary
    total = len(preds)
    acc = (correct / total) * 100 if total > 0 else 0
    avg_time = sum(times) / len(times) if times else 0
    print("\n" + "=" * 60)
    print(f"Finished {total} samples")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Avg time per claim: {avg_time:.2f}s")
    print(f"Errors: {errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()

