"""
Copyright:

  Copyright © 2025 Eric

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file builds a searchable index of the FEVER wikipedia dump.

Code:
"""

import requests
import zipfile
import shutil
import subprocess
import os
import sys
import json
from .utils import (
    WIKI_DIR,
    ZIP_PATH,
    INDEX_DIR,
    PAGES_DIR,
    WIKI_DIR,
)

WIKI_URL = "https://fever.ai/download/fever/wiki-pages.zip"

def load_wiki():
    """
    Downloads the FEVER Wikipedia dump from the URL specified above.
    Extracts the JSONL files, where each record has the fields {id, text, lines}.
    The 'text' field is renamed to 'contents' in place to match Pyserini's expected format.
    """

    if WIKI_DIR.exists():
        # clean
        shutil.rmtree(WIKI_DIR)

    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch and unzip dump
    print("Downloading FEVER Wikipedia dump...")
    res = requests.get(WIKI_URL, stream=True)
    res.raise_for_status()

    with open(ZIP_PATH, "wb") as file:
        for chunk in res.iter_content(chunk_size=8192):
            file.write(chunk)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(WIKI_DIR)

    # Cleans up unused __MACOSX folder
    ZIP_PATH.unlink()
    macos_dir = WIKI_DIR / "__MACOSX"
    if macos_dir.exists():
        shutil.rmtree(macos_dir, ignore_errors=True)

    # Map 'text' field to 'content' in each JSONL row
    print("Preprocessing JSONL files...")
    for file_path in (PAGES_DIR.glob("*.jsonl")):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        with open(file_path, "w", encoding="utf-8") as file:
            for line in lines:
                if not line.strip():
                    continue

                obj = json.loads(line)
                new_obj = {"id": obj["id"], "contents": obj["text"]}
                json.dump(new_obj, file)
                file.write("\n")

def build_index():
    """
    Builds the index from the Wikipedia dump using PySerini, as done in A1.
    """

    if INDEX_DIR.exists():
        return

    # Copying command-line approach from A1. Probably a function for this.
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(PAGES_DIR),
        "--index", str(INDEX_DIR),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storePositions",
        "--storeDocvectors",
        "--storeContents",
    ]

    subprocess.run(cmd, check=True)
    print(f"Index successfully built at {INDEX_DIR}")
