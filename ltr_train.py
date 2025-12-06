import json
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from src.config import INDEX_DIR, DATA_DIR
from datasets import load_dataset
import torch
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import INDEX_DIR, DATA_DIR

from reranker import E2RankReranker, CrossEncoderReranker  
from reranker.weightfunc.consensus_weight import ConsensusWeightFunction
from reranker.weightfunc.credibility_weight import CredibilityWeightFunction
from reranker.weightfunc.temporal_weight import TemporalWeightFunction
from reranker.combine_weights import NeuralCombinerTrainer

def prepare_training_data(max_samples=500, negatives_per_positive=3, batch_size=16):
    print("Loading FEVER dataset...")
    ds = load_dataset("fever", "v1.0", cache_dir=DATA_DIR, split="labelled_dev")
    ds = [
        example for example in ds 
        if example["label"] in ["SUPPORTS", "REFUTES"] and example.get("evidence_wiki_url")
    ]
    ds.reverse()
    print(f"Reversed dataset to train on last {max_samples} valid claims")
    
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(1.2, 0.75)
    
    reranker = E2RankReranker()
    consensus_fn = ConsensusWeightFunction(sim_method="dense")
    credibility_fn = CredibilityWeightFunction(wikipedia_only=True)
    temporal_fn = TemporalWeightFunction(for_fever=True)
    
    all_features = [] 
    claim_data = []  
    valid_claims = 0
    
    batch_claims = []
    batch_doc_pairs = []
    batch_metadata = []
    
    for i, example in enumerate(tqdm(ds, desc="Processing claims")):
        if valid_claims >= max_samples:
            break
        
        claim = example["claim"]
        label = example["label"]
        evidence_url = example.get("evidence_wiki_url")
        
        hits = searcher.search(claim, k=50)
        doc_pairs = []
        for hit in hits:
            doc = searcher.doc(hit.docid)
            if doc:
                doc_pairs.append((hit.docid, doc.contents()))
        
        if not doc_pairs:
            continue
        
        batch_claims.append(claim)
        batch_doc_pairs.append(doc_pairs)
        batch_metadata.append(evidence_url)
        
        # Process batch when full or at end
        if len(batch_claims) == batch_size or (valid_claims + len(batch_claims) >= max_samples):
            # Batch rerank
            reranked_batch = []
            for claim, doc_pairs in zip(batch_claims, batch_doc_pairs):
                reranked = reranker.rerank(claim, doc_pairs, top_k=50)
                reranked_batch.append(reranked)
            
            # Batch compute weights
            for claim, reranked, evidence_url in zip(batch_claims, reranked_batch, batch_metadata):
                consensus_weights = consensus_fn.compute_weights(claim, reranked)
                credibility_weights = credibility_fn.compute_weights(claim, reranked)
                temporal_weights = temporal_fn.compute_weights(claim, reranked)
                
                doc_features = []
                relevant_idx = None
                
                for j, (doc_id, doc_text, rerank_score) in enumerate(reranked):
                    features = np.array([
                        consensus_weights[j],
                        credibility_weights[j],
                        temporal_weights[j],
                        rerank_score
                    ])
                    
                    doc_features.append(features)
                    all_features.append(features)
                    
                    if doc_id == evidence_url:
                        relevant_idx = j
                
                if relevant_idx is not None:
                    claim_data.append({
                        'features': doc_features,
                        'relevant_idx': relevant_idx
                    })
                    valid_claims += 1
            
            # Clear batch
            batch_claims = []
            batch_doc_pairs = []
            batch_metadata = []
    
    print(f"\nCollected {valid_claims} claims with evidence")
    print(f"Total feature vectors: {len(all_features)}")
    
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    for claim in claim_data:
        claim['features'] = scaler.transform(claim['features'])
    
    training_pairs = []
    for claim in claim_data:
        features = claim['features']
        relevant_idx = claim['relevant_idx']
        
        pos_features = features[relevant_idx]
        
        negative_indices = [i for i in range(len(features)) if i != relevant_idx]
        sampled_negatives = random.sample(
            negative_indices,
            min(negatives_per_positive, len(negative_indices))
        )
        
        for neg_idx in sampled_negatives:
            neg_features = features[neg_idx]
            training_pairs.append((pos_features, neg_features))
    
    print(f"Generated {len(training_pairs)} training pairs")
    random.shuffle(training_pairs)
    split_idx = int(0.8 * len(training_pairs))
    train_pairs = training_pairs[:split_idx]
    val_pairs = training_pairs[split_idx:]
    
    print(f"\nTrain:\t{len(train_pairs)} pairs")
    print(f"Val:\t{len(val_pairs)} pairs")
    
    with open("reranker/models/neural_combiner_train.pkl", "wb") as f:
        pickle.dump(train_pairs, f)
    with open("reranker/models/neural_combiner_val.pkl", "wb") as f:
        pickle.dump(val_pairs, f)
    with open("reranker/models/feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved training and validation data in folder reranker/models/")
    
    return train_pairs, val_pairs, scaler

def train_neural_combiner():
    print("Loading training data...")
    with open("reranker/models/neural_combiner_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("reranker/models/neural_combiner_val.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open("reranker/models/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    print(f"Train:\t{len(train_data)} examples")
    print(f"Val:\t{len(val_data)} examples")
    
    n_features = 4
    trainer = NeuralCombinerTrainer(
        n_features=n_features,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer.scaler = scaler
    print(f"Loaded feature scaler")
    
    print(f"\nTraining on {trainer.device}...")
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=100,
        batch_size=128
    )
    
    model_path = "reranker/models/neural_combiner.pt"
    trainer.save_model(str(model_path))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    prepare_training_data(max_samples=500)
    train_neural_combiner()
    