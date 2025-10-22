from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.model_clients import QwenClient
from src.ragar_corag import RagarCorag

if __name__ == "__main__":
    num_samples = 10  # adjust as needed
    ds = load_dataset("fever", "v1.0", trust_remote_code=True)
    split = ds["labelled_dev"].select(range(num_samples))
    fever_labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH EVIDENCE"]

    # Use dummy predictions for now
    preds = []
    labels = []

    # Simulate predictions if you don't have Qwen access
    for i in tqdm(range(len(split))):
        label = split[i]["label"]
        labels.append(label)

        # Example: predict "NOT ENOUGH EVIDENCE" for all
        preds.append("NOT ENOUGH EVIDENCE")

    # Metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=fever_labels, average=None
    )
    pred_counts = Counter(preds)
    label_counts = Counter(labels)

    print(f"\nAccuracy: {accuracy:.3f}")
    print("Per-class Precision:", dict(zip(fever_labels, precision)))
    print("Per-class Recall:", dict(zip(fever_labels, recall)))
    print("Per-class F1:", dict(zip(fever_labels, f1)))
    print("Pred label counts:", pred_counts)
    print("True label counts:", label_counts)
