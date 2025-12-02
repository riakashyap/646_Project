"""
Copyright:

  Copyright © 2025 uchuuronin
  Copyright © 2025 Ria
  Copyright © 2025 Ananya-Jha-code

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
    
class LayerwiseCEKLLoss(nn.Module):
    """
    E2Rank loss: layer-wise cross-entropy + KL divergence.

    Expected model outputs (per forward pass):
        {
            "layer_logits": List[Tensor],   # each tensor: (batch, 2)
            "final_logits": Tensor          # (batch, 2)
        }
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, model_outputs, labels):
        """
        Args:
            model_outputs: dict with:
                - layer_logits: list of tensors (batch, 2)
                - final_logits: tensor (batch, 2)
            labels: tensor of shape (batch,) with {0/1}
        """

        layer_logits: list = model_outputs["layer_logits"]
        final_logits: torch.Tensor = model_outputs["final_logits"]

        num_layers = len(layer_logits)
        if num_layers == 0:
            raise ValueError("model_outputs['layer_logits'] is empty")

        # -----------------------------
        # 1. CE loss on each layer
        # -----------------------------
        ce_losses = []
        for logits in layer_logits:
            ce_losses.append(self.ce(logits, labels))
        L_ce = sum(ce_losses) / num_layers

        # -----------------------------
        # 2. KL divergence: each layer vs final layer
        # -----------------------------
        final_probs = F.softmax(final_logits / self.temperature, dim=-1)

        kl_losses = []
        for logits in layer_logits[:-1]:  # exclude final layer
            layer_log_probs = F.log_softmax(logits / self.temperature, dim=-1)
            kl_losses.append(self.kl(layer_log_probs, final_probs))

        L_kl = sum(kl_losses) / max(1, (num_layers - 1))

        return L_ce + L_kl


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


class PointwiseBCELoss(nn.Module):
    """
    Pointwise binary classification loss on (query, doc) scores.

    Expects:
        - scores: (batch,) or (batch, 1), raw logits
        - labels: (batch,) in {0, 1}
    """

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        # store the scalar, we’ll build the Tensor on the correct device in forward
        self.pos_weight_value = pos_weight

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = scores.view(-1)          # (batch,)
        labels = labels.float().view(-1)  # (batch,)

        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} vs labels {labels.shape}"
            )

        if self.pos_weight_value is not None:
            # pos_weight > 1.0 if positives are rare
            pos_weight = torch.tensor(
                self.pos_weight_value,
                device=scores.device,
                dtype=scores.dtype,
            )
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        return loss_fn(scores, labels)


class PairwiseHingeLoss(nn.Module):
    """
    Margin-based pairwise ranking loss on scalar scores:

        L = max(0, margin - (z_pos - z_neg))

    Encourages: z_pos >= z_neg + margin

    Typically used in ranking / metric learning when you want a hard margin.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.margin = margin
        self.reduction = reduction

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        # (batch,) or (batch, 1) → flatten to (batch,)
        pos_scores = pos_scores.view(-1)
        neg_scores = neg_scores.view(-1)

        if pos_scores.shape != neg_scores.shape:
            raise ValueError(
                f"Shape mismatch: pos_scores {pos_scores.shape} vs neg_scores {neg_scores.shape}"
            )

        # diff = z_pos - z_neg
        diff = pos_scores - neg_scores

        # hinge loss = max(0, margin - diff)
        loss = F.relu(self.margin - diff)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ListwiseCELoss(nn.Module):
    """
    Listwise cross-entropy loss when each query has *exactly one* positive doc.

    Expected:
        scores: (batch, n_docs)
        labels: (batch, n_docs) with exactly one '1' per row.

    This is equivalent to CE over the document list for each query.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Input validation
        if scores.dim() != 2 or labels.dim() != 2:
            raise ValueError(
                f"scores and labels must both be (batch, n_docs), "
                f"got scores={scores.shape}, labels={labels.shape}"
            )

        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} vs labels {labels.shape}"
            )

        # Check one-positive-per-row (optional but helpful)
        if not torch.all(labels.sum(dim=1) == 1):
            raise ValueError(
                "Each row in labels must contain exactly one positive ('1')."
            )

        # Convert one-hot → target index
        targets = labels.argmax(dim=-1)  # (batch,)

        # CE between list scores and positive index
        return self.ce(scores, targets)


LOSS_REGISTRY = {
    "layerwise_ce_kl": LayerwiseCEKLLoss,   # E2Rank
    "pairwise_exp": PairwiseRankingLoss,    # softplus-based
    "pairwise_hinge": PairwiseHingeLoss,    # margin-based
    "pointwise_bce": PointwiseBCELoss,      # BCE on single scores
    "listwise_ce": ListwiseCELoss,          # listwise CE with one positive
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Factory to instantiate a loss by name.

    Example:
        loss_fn = get_loss("pairwise_exp")
        loss_fn = get_loss("pointwise_bce", pos_weight=2.0)
        loss_fn = get_loss("pairwise_hinge", margin=1.0)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss name '{name}'. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name](**kwargs)
