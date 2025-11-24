"""
Copyright:

  Copyright © 2025 uchuuronin
  Copyright © 2025 Ria

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Loss functions for the reranker model training. 
    1. Exponential Pairwise ranking loss (as in Assignment A2)
    2. Layerwise cross-entropy loss with KL divergence (E2Rank approach) 
    
Code:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    
class PairwiseRankingLoss(nn.Module):
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

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'mean' (default), 'sum', or 'none'
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_scores: Tensor of shape (batch,) or (batch, 1)
            neg_scores: Tensor of shape (batch,) or (batch, 1)

        Returns:
            Scalar loss (if reduction is 'mean' or 'sum'),
            otherwise per-example loss tensor.
        """
        # Flatten to 1D to be tolerant of (batch, 1)
        pos_scores = pos_scores.view(-1)
        neg_scores = neg_scores.view(-1)

        if pos_scores.shape != neg_scores.shape:
            raise ValueError(
                f"Shape mismatch: pos_scores {pos_scores.shape} vs neg_scores {neg_scores.shape}"
            )

        # diff = z_pos - z_neg
        diff = pos_scores - neg_scores

        # log(1 + exp(-diff)) = softplus(-diff)
        loss = F.softplus(-diff)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss