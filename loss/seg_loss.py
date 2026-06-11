"""
Dice loss for segmentation.
"""
import torch
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Soft Dice loss over foreground classes.

    Args:
        logits: [B, K, D, H, W] raw logits
        target: [B, D, H, W] integer labels 0..K-1
        smooth: smoothing factor to avoid division by zero

    Returns:
        Scalar loss (1 - mean Dice)
    """
    K = logits.shape[1]
    pred_prob = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target.long(), num_classes=K).permute(0, 4, 1, 2, 3).float()

    # Foreground only (skip background channel 0)
    pred_fg = pred_prob[:, 1:]
    target_fg = target_oh[:, 1:]

    intersection = (pred_fg * target_fg).sum(dim=[2, 3, 4])
    union = pred_fg.sum(dim=[2, 3, 4]) + target_fg.sum(dim=[2, 3, 4])
    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()

def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice + CrossEntropy."""
    return dice_loss(logits, target) + F.cross_entropy(logits, target)