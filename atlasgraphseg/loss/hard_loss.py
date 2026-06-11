"""
Hard Prior Loss — penalize anatomically impossible adjacencies.

Uses a binary atlas prior to define which ROI pairs CAN be adjacent.
Any predicted adjacency on an impossible edge gets penalized.
"""
import torch
import torch.nn as nn
from atlasgraphseg.structural_prior import get_priors_for_num_classes

class HardPriorLoss(nn.Module):
    """
    Hard Structural Prior Loss
    해부학적으로 불가능한 연결(Impossible)에 페널티를 부여합니다.
    """
    def __init__(self, num_classes, enforce_connectivity=False, penalty_weight=1.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.enforce_connectivity = enforce_connectivity
        
        prior_mask, _ = get_priors_for_num_classes(num_classes)
        
        self.register_buffer('prior_mask', prior_mask)
        self.register_buffer('impossible_mask', 1 - prior_mask)
        
        C = prior_mask.shape[0]
        self.register_buffer('diag_mask', 1 - torch.eye(C))
        
        # impossible edge 개수 (정규화 상수)
        n_impossible = (self.impossible_mask * self.diag_mask).sum().item()
        self.n_impossible = max(n_impossible, 1.0)
        
        n_possible = (prior_mask * (1 - torch.eye(C))).sum().item()
        self.n_possible = max(n_possible, 1.0)
        
        print(f"  HardPriorLoss: {int(n_possible)//2} possible edges, "
              f"{int(n_impossible)//2} impossible edges")


    def forward(self, A_pred):
        """
        Args:
            A_pred: [B, C, C] Predicted Soft Adjacency (C = num_classes - 1)
        """
        # 안전장치: 차원 확인
        if A_pred.shape[-1] != self.prior_mask.shape[-1]:
            raise RuntimeError(
                f"Dimension mismatch! A_pred: {A_pred.shape}, "
                f"Prior: {self.prior_mask.shape}"
            )

        # Frobenius norm 정규화
        fro_norm = A_pred / (torch.linalg.norm(A_pred, ord='fro', dim=(1,2)).clamp_min(1e-8).view(-1,1,1)) # Frobenius 정규화
        A_norm = fro_norm # [B, C, C]


        # ── Negative Constraint: impossible edge만 penalize ──
        # impossible_mask * diag_mask: 자기자신 제외한 불가능 엣지만
        violation = A_norm * self.impossible_mask * self.diag_mask
        # impossible edge 수로 나눠서 edge 밀도에 무관하게 스케일링
        loss_not_exist = violation.sum(dim=[1, 2]).mean()
        
        # ── Positive Constraint (옵션) ──
        loss_exist = A_pred.new_zeros(()) 
        if self.enforce_connectivity:
            consistency = self.prior_mask * (1 - A_norm) * self.diag_mask
            loss_exist = consistency.sum(dim=[1, 2]).mean() / self.n_possible
            
        return loss_exist + self.penalty_weight * loss_not_exist