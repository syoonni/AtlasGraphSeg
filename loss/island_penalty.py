"""
Island Penalty — differentiable regularization against disconnected components.

Encourages each foreground class to form spatially contiguous regions
by penalizing sharp probability transitions within confident regions.
"""
import torch
import torch.nn as nn

class IslandPenaltyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, prob):
        """
        prob: [B, K, D, H, W]
        
        Vectorized: 모든 foreground 채널을 한번에 처리.
        기존: for c in range(1, K) 루프 → 78번 반복
        개선: 텐서 연산으로 루프 제거
        """
        # Skip background (channel 0)
        P = prob[:, 1:]  # [B, C, D, H, W], C = K-1

        # Spatial gradients along each axis (all channels at once)
        grad_d = (P[:, :, 1:] - P[:, :, :-1]).abs()      # [B, C, D-1, H, W]
        grad_h = (P[:, :, :, 1:] - P[:, :, :, :-1]).abs() # [B, C, D, H-1, W]
        grad_w = (P[:, :, :, :, 1:] - P[:, :, :, :, :-1]).abs() # [B, C, D, H, W-1]

        # Penalty: gradient * confidence at both ends
        term_d = (grad_d * P[:, :, 1:] * P[:, :, :-1]).mean()
        term_h = (grad_h * P[:, :, :, 1:] * P[:, :, :, :-1]).mean()
        term_w = (grad_w * P[:, :, :, :, 1:] * P[:, :, :, :, :-1]).mean()

        return self.weight * (term_d + term_h + term_w)