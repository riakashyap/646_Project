FABLE is a multi-agent [CoRAG](https://arxiv.org/abs/2404.12065) pipeline for automated fact verification. It is evaluated on the [FEVER](https://fever.ai/dataset/fever.html) dataset. Our goal was to experiment with how incorporating reranking and multi-agent debates can improve the ability of agentic RAG systems to verify real-world claims. 

# Installation
The repository is written and tested for Python 3.11. Start by creating a fresh [conda](https://anaconda.org/channels/anaconda/packages/conda/overview) environment:
```bash
conda create -n project-646 python=3.11
conda activate project-646
pip install -r requirements.txt
```

Next, install [llama.cpp](https://github.com/ggml-org/llama.cpp). This tool is used to interface with models hosted locally. Models must be in GGUF format and can be downloaded from [Hugging Face](https://huggingface.co/models). Note if using a GPU, compile llama.cpp with CUDA or Vulkan support. 

The reranker component and accompanying tests require [DeBERTaV3](https://huggingface.co/naver/trecdl22-crossencoder-debertav3), which can be downloaded by running the shell script:
```bash
bash reranker/download_hf_model.sh
```

The repository also requires [Java 21](https://www.oracle.com/java/technologies/downloads/#java21) due to a dependency on [Pyserini](https://github.com/castorini/pyserini).

Finally, to validate the installation and build the search index for FEVER's 2017 Wikipedia dump (which is required for the supplied `main.py` evaluation script to function), run the test suite:
```
python run_tests.py
```

# Usage
Start the llama.cpp server on port 4568, hosting your desired model:
```bash
llama-server --reasoning-budget 0 --port 4568 -t 8 -m /path/to/your/model.gguf
```

`main.py` runs the pipeline on FEVER claims. This script supports a variety of command-line arguments. For instance, to run the pipeline with our custom prompts, verdict debating, and reranking:
```bash
python main.py --debate-verdict --reranker
```

To view the entire list of options:
```bash
python main.py --help
```