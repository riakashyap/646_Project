import requests
import zipfile
import shutil
import subprocess
import os
import json
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data"
wiki_dir = data_dir / "wiki"
zip_path = data_dir / "wiki-pages.zip"
index_dir = wiki_dir / "index"
pages_dir = wiki_dir / "wiki-pages"
wiki_url = "https://fever.ai/download/fever/wiki-pages.zip"

def load_wiki():
    """
    Downloads the FEVER Wikipedia dump from the URL specified above.
    Extracts the JSONL files, where each record has the fields {id, text, lines}.
    The 'text' field is renamed to 'contents' in place to match Pyserini's expected format.
    """

    if pages_dir.exists() and os.listdir(pages_dir):
        print(f"Skipping wiki download, using existing data at: {pages_dir}")
        return

    wiki_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and unzip dump
    print("Downloading FEVER Wikipedia dump...")
    res = requests.get(wiki_url, stream=True)
    res.raise_for_status()

    with open(zip_path, "wb") as file:
        for chunk in res.iter_content(chunk_size=8192):
            file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(wiki_dir)

    # Cleans up unused __MACOSX folder
    zip_path.unlink()
    macos_dir = wiki_dir / "__MACOSX"
    if macos_dir.exists():
        shutil.rmtree(macos_dir, ignore_errors=True)

    # Map 'text' field to 'content' in each JSONL row
    print("Preprocessing JSONL files...")
    for file_path in (pages_dir.glob("*.jsonl")):
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

    print(f"Files ready at: {pages_dir}")

def build_index():
    """
    Builds the index from the Wikipedia dump using PySerini, as done in A1.
    """

    if index_dir.exists() and os.listdir(index_dir):
        print(f"Skipping index build, using existing index at: {index_dir}")
        return

    if not pages_dir.exists() or not os.listdir(pages_dir):
        load_wiki()

    # Copying command-line approach from A1. Probably a function for this.
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(pages_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw"
    ]

    subprocess.run(cmd, check=True)
    print(f"Index successfully built at {index_dir}")

if __name__ == "__main__":
    build_index()

    # Test retrieval
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(1.2, 0.75)
    query = "Who was the first president of the United States?"
    hits = searcher.search(query, k=5)
    for hit in hits:
        doc = searcher.doc(hit.docid)
        print(f"{hit.docid} ({hit.score:.3f})")
