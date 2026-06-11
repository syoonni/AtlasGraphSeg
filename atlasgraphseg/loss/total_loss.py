"""
Loss orchestration — computes each loss component independently.

Does NOT sum losses internally. Returns individual scalars so that
the training loop can apply lambda weights externally.

Components:
    1. seg_loss    : Dice + CE  (from seg_loss.py)
    2. hard_loss   : Atlas impossible adjacency penalty  (from hard_loss.py)
    3. island_loss : Disconnected component penalty  (from island_penalty.py)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .seg_loss import segmentation_loss, dice_loss
from .hard_loss import HardPriorLoss
from .island_penalty import IslandPenaltyLoss
from atlasgraphseg.graph import soft_adjacency_from_probs, hard_adjacency_from_mask


def compute_all_losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    connectivity: str = '6-connectivity',
    hard_loss_fn: Optional[HardPriorLoss] = None,
    island_loss_fn: Optional[IslandPenaltyLoss] = None,
    is_train: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute each loss component as an independent scalar.

    Does NOT apply lambda weights or sum anything — that's the caller's job.

    Args:
        logits: [B, K, D, H, W] raw model output
        target: [B, D, H, W] integer labels 0..K-1
        num_classes: total classes including background
        connectivity: '6-connectivity', '18-connectivity', or '26-connectivity'
        hard_loss_fn: pre-initialized HardPriorLoss (None = skip)
        island_loss_fn: pre-initialized IslandPenaltyLoss (None = skip, eval-only)
        is_train: whether we're in training mode (if True, island_loss is skipped)

    Returns:
        Dict of individual loss scalars:
            'seg'    : Dice + CE                          (for backprop)
            'hard'   : atlas impossible adjacency penalty (for backprop)
            'island' : disconnected component penalty     (eval-only metric)
        Plus diagnostics:
            'A_pred_sum', 'A_gt_sum'
    """
    device = logits.device
    zero = torch.tensor(0.0, device=device)
    
    # ── 1. Segmentation loss (Dice + CE) ──
    L_seg = segmentation_loss(logits, target)
    
    # ── Shared: adjacency matrices ──
    with torch.amp.autocast('cuda', enabled=False):  # hard_loss는 FP32로 계산하는 게 안정적
        prob = F.softmax(logits, dim=1)
        A_pred = soft_adjacency_from_probs(prob, connectivity)
        A_gt = hard_adjacency_from_mask(target, num_classes, connectivity)

    
        # ── 2. Hard prior loss ──
        if hard_loss_fn is not None and A_pred.sum() > 0:
            L_hard = hard_loss_fn(A_pred)
        else:
            L_hard = zero
    
        # ── 3. Island penalty ──
        if island_loss_fn is not None and not is_train: # is_train=True일 때는 skip
            L_island = island_loss_fn(prob)
        else:
            L_island = zero
    
    return {
        'seg': L_seg,
        'hard': L_hard,
        'island': L_island,
        # diagnostics
        'A_pred_sum': A_pred.sum().detach(),
        'A_gt_sum': A_gt.sum().detach(),
    }
