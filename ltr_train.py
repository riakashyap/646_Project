import json
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from reranker.e2rank_reranker import E2RankReranker
from reranker.weightfunc.consensus_weight import ConsensusWeightFunction
from reranker.weightfunc.credibility_weight import CredibilityWeightFunction
from reranker.weightfunc.temporal_weight import TemporalWeightFunction
from src.config import INDEX_DIR, DATA_DIR
from datasets import load_dataset
import torch
from reranker.combine_weights import NeuralCombinerTrainer
import pickle


def prepare_training_data(max_samples=500):
    print("Loading FEVER dataset...")
    ds = load_dataset("fever", "v1.0", cache_dir=DATA_DIR, split="labelled_dev")
    
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(1.2, 0.75)
    
    reranker = E2RankReranker(reranking_block_map={8: 50, 16: 28, 24: 10})
    consensus_fn = ConsensusWeightFunction(sim_method="dense")
    credibility_fn = CredibilityWeightFunction(wikipedia_only=True)
    temporal_fn = TemporalWeightFunction(for_fever=True)
    
    training_examples = []
    
    for i, example in enumerate(ds):
        if i >= max_samples:
            break
        
        claim = example["claim"]
        label = example["label"]
        evidence_url = example.get("evidence_wiki_url")
        
        if label not in ["SUPPORTS", "REFUTES"] or not evidence_url:
            continue
        
        hits = searcher.search(claim, k=50)
        doc_pairs = []
        for hit in hits:
            doc = searcher.doc(hit.docid)
            if doc:
                doc_pairs.append((hit.docid, doc.contents()))
        
        if not doc_pairs:
            continue
        
        reranked = reranker.rerank(claim, doc_pairs, top_k=50)
        
        consensus_weights = consensus_fn.compute_weights(claim, reranked)
        credibility_weights = credibility_fn.compute_weights(claim, reranked)
        temporal_weights = temporal_fn.compute_weights(claim, reranked)
        
        for j, (doc_id, doc_text, rerank_score) in enumerate(reranked):
            # Features: [consensus_w, credibility_w, temporal_w, rerank_score]
            features = np.array([
                consensus_weights[j],
                credibility_weights[j],
                temporal_weights[j],
                rerank_score
            ])
            
            # Label: 1 if this doc contains evidence, 0 otherwise
            is_relevant = 1 if doc_id == evidence_url else 0
            
            training_examples.append((features, is_relevant))
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{max_samples} claims, {len(training_examples)} examples")
    
    # Split 80/20 train/val
    split_idx = int(0.8 * len(training_examples))
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]
    
    print(f"\n\tTotal: {len(training_examples)} examples")
    print(f"\tTrain: {len(train_data)}")
    print(f"\tVal:   {len(val_data)}")
    
    with open("reranker/models/neural_combiner_train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open("reranker/models/neural_combiner_val.pkl", "wb") as f:
        pickle.dump(val_data, f)
    print(f"Saved training and validation data in folder reranker/models/")
    
    return train_data, val_data


def train_neural_combiner():
    print("Loading training data...")
    with open("reranker/models/neural_combiner_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("reranker/models/neural_combiner_val.pkl", "rb") as f:
        val_data = pickle.load(f)
    
    print(f"Train:\t{len(train_data)} examples")
    print(f"Val:\t{len(val_data)} examples")
    
    n_features = 4
    trainer = NeuralCombinerTrainer(
        n_features=n_features,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Training on {trainer.device}...")
    trainer.train(
        train_data=train_data,  
        val_data=val_data,
        epochs=100,
        batch_size=64
    )
    
    model_path = "reranker/models/neural_combiner.pt"
    trainer.save_model(str(model_path))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_data, val_data = prepare_training_data(max_samples=500)
    train_neural_combiner()
    