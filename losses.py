"""
Custom loss functions for handling class imbalance
- PolyFocalLoss: Focal Loss + Polynomial adjustment term
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class PolyFocalLoss(nn.Module):
    """
    Poly-Focal Loss: Focal Loss with polynomial modulation.

    Combines:
    1. Focal Loss (gamma): down-weights easy examples, focuses on hard ones
    2. Poly term (epsilon): boosts gradient for ambiguous samples
    3. Class weights (alpha): compensates for class imbalance

    Reference: PolyLoss (Leng et al., 2022) adapted with Focal Loss base.

    Args:
        gamma: Focal loss focusing parameter. Higher = more focus on hard examples (default: 2.0)
        epsilon: Polynomial coefficient. Controls extra gradient for uncertain predictions (default: 1.0)
        alpha: Per-class weights tensor. If None, all classes weighted equally.
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, gamma=2.0, epsilon=1.0, alpha=None, reduction='mean'):
        super(PolyFocalLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
            else:
                self.alpha = None
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits: Model output (B, C) - raw logits before softmax
            targets: Ground truth labels (B,) - class indices
        """
        num_classes = logits.size(1)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Get probability of correct class: p_t
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Cross entropy: -log(p_t)
        ce_loss = -torch.log(p_t.clamp(min=1e-8))

        # Focal loss
        focal_loss = focal_weight * ce_loss

        # Poly term: epsilon * (1 - p_t)
        # This adds extra gradient signal for ambiguous predictions
        poly_term = self.epsilon * (1.0 - p_t)

        # Combined loss per sample
        loss = focal_loss + poly_term

        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]  # (B,)
            loss = alpha_t * loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_class_weights(labels, method='inverse_freq'):
    """
    Compute class weights from label distribution.

    Args:
        labels: List of integer labels
        method:
            'inverse_freq' - weight = total / (num_classes * class_count)
            'effective_num' - Effective Number of Samples (Cui et al., 2019), beta=0.999

    Returns:
        weights: torch.Tensor of shape (num_classes,)
    """
    counter = Counter(labels)
    num_classes = len(counter)
    total = len(labels)

    if method == 'inverse_freq':
        # Inverse frequency: w_i = N / (C * n_i)
        weights = []
        for i in range(num_classes):
            count = counter[i]
            w = total / (num_classes * count)
            weights.append(w)
        weights = torch.tensor(weights, dtype=torch.float32)

    elif method == 'effective_num':
        # Effective Number of Samples: w_i = (1 - beta) / (1 - beta^n_i)
        beta = 0.999
        weights = []
        for i in range(num_classes):
            count = counter[i]
            effective_num = 1.0 - beta ** count
            w = (1.0 - beta) / effective_num
            weights.append(w)
        weights = torch.tensor(weights, dtype=torch.float32)
        # Normalize so weights sum to num_classes
        weights = weights / weights.sum() * num_classes

    else:
        raise ValueError(f"Unknown method: {method}")

    return weights
