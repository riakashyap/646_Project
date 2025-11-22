"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Loss functions for the reranker model training. 
    1. Exponential Pairwise ranking loss (as in Assignment A2)
    2. Layerwise cross-entropy loss with KL divergence (E2Rank approach) 
    
Code:
"""

class LayerwiseCEKLLoss():
    """
    E2Rank loss: Layerwise cross-entropy + KL divergence. (pg3 of paper for reference)
    
    L_total = L_layerwise + L_divergence
    
    Where:
    - L_layerwise = (1/|L|) * SUM_OF CE(y_l, y) over layers
    where y_l is the probability distribution at the l-th layer, 
    and y is the target one,
    
    - L_divergence = (1/(|L|-1)) * SUM_OF KL(y_l || y_final) over intermediate layers
    where y_l is the probability distribution at the l-th layer (student) 
    where y_l is the probability distribution at the l-th layer (teacher)

    Goal:
    1. Each layer should be able to rank positive doc highest (via CE)
    2. Intermediate layers should be consistent with final layer (via KL)
    """
    
    
class PairwiseRankingLoss():
    """
    Pairwise ranking loss with exponential penalty (from Assignment 2).
    
    L_exp = (1/|D|) * SUM_OF log(1 + exp(-(z_q,d+ - z_q,d-)))
    
    Where:
    - z_q,d+ = Cross-encoder score for relevant document d+ given query q
    - z_q,d- = Cross-encoder score for non-relevant document d- given query q
    - D = Set of (query, positive_doc, negative_doc) triplets
    
    Goal:
    Encourages positive score to exceed negative score (z_q,d+ > z_q,d-).
    It is smooth and places higher penalty when z_q,d+ < z_q,d-
    """