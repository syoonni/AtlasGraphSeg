"""
Pure PyTorch fine-tuning of a 3D segmentation model with an ROI-graph prior loss.
- No TensorFlow, no MONAI dependency.
- Includes a compact 3D U-Net implemented in PyTorch.
- Builds a differentiable (soft) ROI adjacency from predicted per-voxel probabilities.
- Matches it to a target adjacency from ground-truth or atlas masks.

Assumptions
- Labels are in 0..K-1 where 0 = background and 1..K-1 = ROI classes.
- Inputs: (B, C_in=1, D, H, W); labels: (B, D, H, W) int.
- Default connectivity is 6-neighborhood (face contact).

Note
- For 256^3 volumes, train patch-wise (e.g., 96^3) for memory.
"""
from __future__ import annotations
import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# 3D U-Net (pure PyTorch)
###############################################

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='in'):  # 'in' = InstanceNorm, 'bn' = BatchNorm
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == 'in' else nn.BatchNorm3d
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_ch: int = 32, norm: str='in'):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch, norm)
        self.enc2 = DoubleConv(base_ch, base_ch*2, norm)
        self.enc3 = DoubleConv(base_ch*2, base_ch*4, norm)
        self.enc4 = DoubleConv(base_ch*4, base_ch*8, norm)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(base_ch*8, base_ch*16, norm)

        self.up4 = nn.ConvTranspose3d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch*16, base_ch*8, norm)
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch*8, base_ch*4, norm)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch*4, base_ch*2, norm)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch*2, base_ch, norm)

        self.head = nn.Conv3d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        return logits

###############################################
# Connectivity & shifting utilities
###############################################

def connectivity_offsets(connectivity: int = 6) -> List[Tuple[int,int,int]]:
    assert connectivity in (6, 18, 26)
    offs: List[Tuple[int,int,int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                md = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and md == 1:
                    offs.append((dx, dy, dz))
                elif connectivity == 18 and (md == 1 or (md == 2 and max(abs(dx),abs(dy),abs(dz))==1)):
                    offs.append((dx, dy, dz))
                elif connectivity == 26:
                    offs.append((dx, dy, dz))
    return offs


def shift3d_pad(x: torch.Tensor, dx: int, dy: int, dz: int) -> torch.Tensor:
    """Shift a 5D tensor [B,C,D,H,W] by (dx,dy,dz) with zero padding (no wrap)."""
    assert x.ndim == 5
    B,C,D,H,W = x.shape
    x0, x1 = max(0, dx), D + min(0, dx)
    y0, y1 = max(0, dy), H + min(0, dy)
    z0, z1 = max(0, dz), W + min(0, dz)
    xs0, xs1 = x0 - dx, x1 - dx
    ys0, ys1 = y0 - dy, y1 - dy
    zs0, zs1 = z0 - dz, z1 - dz
    out = x.new_zeros((B,C,D,H,W))
    if x1 > x0 and y1 > y0 and z1 > z0:
        out[:, :, x0:x1, y0:y1, z0:z1] = x[:, :, xs0:xs1, ys0:ys1, zs0:zs1]
    return out

###############################################
# Adjacency builders (soft from probs, hard from masks)
###############################################

def soft_adjacency_from_probs(
    prob: torch.Tensor,
    connectivity: int = 6,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable ROI adjacency from per-voxel probabilities.
    Args:
        prob: [B, K, D, H, W] softmax probabilities (K includes background at channel 0).
    Returns:
        A: [B, K-1, K-1] soft, symmetric adjacency matrix per batch.
    """
    assert prob.ndim == 5, "prob must be [B,K,D,H,W]"
    B, K, D, H, W = prob.shape
    if K <= 1:
        raise ValueError("Need at least background + 1 ROI class.")
    P = prob[:, 1:, ...]  # [B,C,D,H,W], C=K-1 (exclude background)
    C = P.shape[1]

    offs = connectivity_offsets(connectivity)
    A = prob.new_zeros((B, C, C))
    for (dx, dy, dz) in offs:
        P_shift = shift3d_pad(P, dx, dy, dz)
        inter = torch.einsum('bcxyz,bkxyz->bck', P, P_shift)  # sum over spatial dims
        A = A + inter
    A = 0.5 * (A + A.transpose(1,2))
    A = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
    A = torch.clamp(A, min=0.0) + epsilon
    return A


def hard_adjacency_from_mask(
    mask: torch.Tensor,
    num_classes: int,
    connectivity: int = 6,
) -> torch.Tensor:
    """
    Hard adjacency from integer label mask using 6/18/26-neighborhood.
    Args:
        mask: [B, D, H, W] integer labels in 0..num_classes-1 (0=background)
    Returns:
        A: [B, num_classes-1, num_classes-1]
    """
    assert mask.ndim == 4, "mask must be [B,D,H,W]"
    onehot = F.one_hot(mask.long(), num_classes=num_classes)  # [B,D,H,W,K]
    onehot = onehot.permute(0,4,1,2,3).contiguous()          # [B,K,D,H,W]
    P = onehot[:, 1:, ...].float()                           # [B,C,D,H,W]

    offs = connectivity_offsets(connectivity)
    B, C = P.shape[:2]
    A = mask.new_zeros((B, C, C), dtype=torch.float32)
    for (dx, dy, dz) in offs:
        P_shift = shift3d_pad(P, dx, dy, dz)
        inter = torch.einsum('bcxyz,bkxyz->bck', P, P_shift)
        A = A + inter
    A = 0.5 * (A + A.transpose(1,2))
    A = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
    return A


def normalize_adjacency(A: torch.Tensor, method: str = 'fro', eps: float = 1e-8) -> torch.Tensor:
    """Normalize adjacency for stable loss ('fro' or 'row')."""
    if method == 'fro':
        n = torch.linalg.norm(A, ord='fro', dim=(1,2)).clamp_min(eps).view(-1,1,1)
        return A / n
    elif method == 'row':
        s = A.sum(dim=2, keepdim=True).clamp_min(eps)
        return A / s
    else:
        return A


def graph_prior_loss(
    A_pred: torch.Tensor,
    A_tgt: torch.Tensor,
    norm: str = 'fro',
    p: int = 1,
    edge_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Distance between predicted and target adjacencies."""
    Ap = normalize_adjacency(A_pred, method=norm)
    At = normalize_adjacency(A_tgt, method=norm)
    diff = (Ap - At).abs() if p == 1 else (Ap - At).pow(2)
    if edge_mask is not None:
        diff = diff * edge_mask
    loss = diff.view(diff.shape[0], -1).mean(dim=1).mean()
    return loss

###############################################
# Dice + Cross-Entropy (pure PyTorch)
###############################################

def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5, ce_weight: float = 1.0) -> torch.Tensor:
    """
    Multi-class Dice + CrossEntropy.
    - logits: [B,K,D,H,W]
    - target: [B,D,H,W] with values in 0..K-1
    """
    B, K = logits.shape[:2]
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target.long(), num_classes=K).permute(0,4,1,2,3).float()

    # Dice over classes 1..K-1 (exclude background)
    probs_fg = probs[:, 1:]
    onehot_fg = onehot[:, 1:]
    dims = (0,2,3,4)
    inter = (probs_fg * onehot_fg).sum(dim=dims)
    den = probs_fg.sum(dim=dims) + onehot_fg.sum(dim=dims)
    dice = (2*inter + smooth) / (den + smooth)
    dice_loss = 1 - dice.mean()

    ce_loss = F.cross_entropy(logits, target)
    return dice_loss + ce_weight * ce_loss

###############################################
# Training step & loop
###############################################

def training_step(
    batch: dict,
    model: nn.Module,
    lambda_graph: float = 0.1,
    connectivity: int = 6,
    num_classes: int = 5,
    use_atlas_prior: bool = True,
) -> Tuple[torch.Tensor, dict]:
    device = next(model.parameters()).device
    image = batch['image'].to(device)
    logits = model(image)

    losses = {}

    # Segmentation loss (if GT present)
    if 'label' in batch:
        label = batch['label'].to(device)
        loss_seg = dice_ce_loss(logits, label)
    else:
        loss_seg = logits.new_tensor(0.0)
    losses['seg'] = float(loss_seg.detach().cpu())

    # Graph prior loss
    prob = F.softmax(logits, dim=1)
    A_pred = soft_adjacency_from_probs(prob, connectivity=connectivity)

    targets = []
    if use_atlas_prior and ('atlas' in batch):
        A_atlas = hard_adjacency_from_mask(batch['atlas'].to(device), num_classes=num_classes, connectivity=connectivity)
        targets.append(A_atlas)
    if 'label' in batch:
        A_gt = hard_adjacency_from_mask(batch['label'].to(device), num_classes=num_classes, connectivity=connectivity)
        targets.append(A_gt)

    if len(targets) == 0:
        loss_graph = logits.new_tensor(0.0)
    else:
        A_tgt = torch.stack(targets, dim=0).mean(dim=0)
        loss_graph = graph_prior_loss(A_pred, A_tgt, norm='fro', p=1)
    losses['graph'] = float(loss_graph.detach().cpu())

    total = loss_seg + lambda_graph * loss_graph
    losses['total'] = float(total.detach().cpu())
    return total, losses


def train_loop(train_loader,
               num_classes: int = 5,
               epochs: int = 10,
               lr: float = 1e-4,
               lambda_graph: float = 0.1,
               connectivity: int = 6,
               device: str = 'cuda',
               in_channels: int = 1,
               base_ch: int = 32):
    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base_ch=base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running = {'seg': 0.0, 'graph': 0.0, 'total': 0.0}
        n = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_step(batch, model,
                                        lambda_graph=lambda_graph,
                                        connectivity=connectivity,
                                        num_classes=num_classes,
                                        use_atlas_prior=True)
            loss.backward()
            optimizer.step()
            for k in running:
                running[k] += parts.get(k, 0.0)
            n += 1
        print(f"Epoch {epoch}: seg={running['seg']/max(1,n):.4f}, graph={running['graph']/max(1,n):.4f}, total={running['total']/max(1,n):.4f}")
    return model

###############################################
# Demo with random tensors (sanity check)
###############################################
if __name__ == '__main__':
    class RandomIter:
        def __init__(self, steps=3):
            self.steps = steps
            self.i = 0
        def __iter__(self):
            return self
        def __next__(self):
            if self.i >= self.steps:
                raise StopIteration
            self.i += 1
            B, K = 1, 5  # 1 background + 4 ROIs
            D,H,W = 64,64,64
            img = torch.randn(B,1,D,H,W)
            label = torch.zeros(B,D,H,W, dtype=torch.long)
            label[:,  8:28,  8:40,  8:40] = 1
            label[:, 28:48,  8:40,  8:40] = 2
            label[:,  8:28, 40:56,  8:40] = 3
            label[:, 16:40, 16:48, 32:56] = 4
            atlas = label.clone()  # demo only
            return {'image': img, 'label': label, 'atlas': atlas}

    loader = list(RandomIter(steps=4))
    train_loop(loader, num_classes=5, epochs=100, lr=1e-4, lambda_graph=0.2, connectivity=6, device='cuda', base_ch=16)