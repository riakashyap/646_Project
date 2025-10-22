from pyserini.search.lucene import LuceneSearcher
from datasets import load_dataset
from collections import Counter
from src.index import INDEX_DIR
from tqdm import tqdm
import glob
import os

if __name__ == "__main__":
    num_samples = 50  # adjust as needed

    # Load all JSON files from your wiki-pages folder
    wiki_folder = r"D:\CS646\mistral\wiki-pages\wiki-pages"
    json_files = glob.glob(os.path.join(wiki_folder, "*.jsonl"))

    # Load as a dataset
    ds = load_dataset("json", data_files=json_files)
    # Suppose the claims are in 'labelled_dev' subset — adjust if needed
    split = ds["train"].select(range(num_samples))

    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=1.2, b=0.75)

    fever_labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH EVIDENCE"]
    labels = []
    preds = []
    retrieved_docs = []

    print(f"Evaluating BM25 on {num_samples} FEVER samples...")

    for i in tqdm(range(len(split))):
        claim = split[i]["claim"]
        label = split[i].get("label", "NOT ENOUGH EVIDENCE")
        labels.append(label)

        hits = searcher.search(claim, k=3)
        if len(hits) == 0:
            preds.append("NOT ENOUGH EVIDENCE")
        else:
            top_doc = searcher.doc(hits[0].docid).get("contents").lower()
            if any(word in top_doc for word in claim.lower().split()):
                preds.append("SUPPORTS")
            else:
                preds.append("REFUTES")

        retrieved_docs.append([searcher.doc(h.docid).get("contents") for h in hits])

    accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    pred_counts = Counter(preds)
    label_counts = Counter(labels)

    print(f"\nAccuracy: {accuracy:.3f}")
    print("Pred labels:", pred_counts)
    print("True labels:", label_counts)
