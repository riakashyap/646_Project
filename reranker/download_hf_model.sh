#!/usr/bin/env bash

LOCAL_DIR="./models/naver-trecdl22-crossencoder-debertav3"
URL="https://huggingface.co/naver/trecdl22-crossencoder-debertav3/resolve/main"

mkdir -p "$LOCAL_DIR"

wget -P "$LOCAL_DIR" \
    "$URL/config.json" \
    "$URL/pytorch_model.bin" \
    "$URL/tokenizer_config.json" \
    "$URL/vocab.txt" \
    "$URL/special_tokens_map.json" \
    "$URL/tokenizer.json"

echo "Model saved to $LOCAL_DIR"

## If you have huggingface-cli configured, you can uncomment below instead:
# huggingface-cli download naver/trecdl22-crossencoder-debertav3 \
#     --local-dir "$LOCAL_DIR" \
#     --local-dir-use-symlinks False
