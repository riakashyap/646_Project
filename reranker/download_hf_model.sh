#!/usr/bin/env bash

LOCAL_DIR="reranker/models/naver/trecdl22-crossencoder-debertav3"
URL="https://huggingface.co/naver/trecdl22-crossencoder-debertav3"

mkdir -p "$LOCAL_DIR"

## Using huggingface-cli to download the model
huggingface-cli download "naver/trecdl22-crossencoder-debertav3" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False

echo "Model saved to $LOCAL_DIR"

## If you don't have huggingface-cli configured, you can uncomment below instead:
# wget -P "$LOCAL_DIR" \
#     "$URL/config.json" \
#     "$URL/pytorch_model.bin" \
#     "$URL/tokenizer_config.json" \
#     "$URL/vocab.txt" \
#     "$URL/special_tokens_map.json" \
#     "$URL/tokenizer.json"

