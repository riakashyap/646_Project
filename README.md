# 646_Project
This repository contains the implementation of claim verification systems for the COMPSCI 646 project, including:
- **Baseline**: RAGAR (Retrieval-Augmented Generation with Active Reasoning)
- **Enhancement**: MADR (Multi-Agent Debate Refinement)

## Quick Start

### 1. Start the LLM Server
```bash
llama-server -hf MaziyarPanahi/Qwen3-4B-GGUF:Q5_K_M --host 127.0.0.1 --port 4568
```

### 2. Run Claim Verification

**Baseline (single agent):**
```bash
python main.py -n 10 -t -r
```

**MADR (multi-agent debate):**
```bash
python main.py -n 10 -t -r --madr --num-agents 3 --debate-rounds 3