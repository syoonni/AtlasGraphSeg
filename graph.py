from typing import Dict, List, Tuple, Optional, Literal

import torch
import torch.nn.functional as F

from graph_prior import connectivity_offsets, shift3d_pad, validate_tensor_shapes

def soft_adjacency_from_probs(
    prob: torch.Tensor,
    connectivity: Literal['6-connectivity', '18-connectivity', '26-connectivity'] = '6-connectivity',
    epsilon: float = 1e-6,
) -> torch.Tensor:
    
    validate_tensor_shapes(prob)

    B, K, D, H, W = prob.shape
    if K <= 1:
        raise ValueError("Number of classes K must be greater than 1 to compute adjacency.")
    
    # Exclude background channel (index 0)
    P = prob[:, 1:, :, :, :]  # Shape: [B, C, D, H, W] where C = K-1
    C = P.shape[1]

    # Get neighborhood offsets
    offsets = connectivity_offsets(connectivity)

    # Initialize adjacency matrix
    A = prob.new_zeros((B, C, C))

    # Compute adjacency by summing interactions across all offsets
    for (dx, dy, dz) in offsets:
        P_shift = shift3d_pad(P, dx, dy, dz) # Shifted probability map
        # Compute pairwise interactions and sum over spatial dimensions
        interaction = torch.einsum("bcxyz, bkxyz -> bck", P, P_shift)
        A = A + interaction

    # Make symmetric and remove diagonal
    A = 0.5 * (A + A.transpose(1, 2))
    A = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))

    return A

def hard_adjacency_from_mask(
    mask: torch.Tensor,
    num_classes: int, 
    connectivity: Literal['6-connectivity', '18-connectivity', '26-connectivity'] = '6-connectivity',
) -> torch.Tensor:
    """
    Hard adjacency from integer label mask using 6/18/26-neighborhood.
    
    Args:
        mask: [B, D, H, W] integer labels in 0..num_classes-1 (0=background)
        num_classes: Total number of classes INCLUDING background
        connectivity: Neighborhood definition
        
    Returns:
        A: [B, C, C] where C = num_classes-1 (excluding background)
    """
    assert mask.dim() == 4, f"Mask must be a 4D [B, D, H, W], got {mask.dim()}D"

    # Get actual max label in mask
    max_label = mask.max().item()
    
    # Use the larger of num_classes and max_label+1 to avoid index errors
    onehot_classes = max(num_classes, max_label + 1)
    
    # Convert to one-hot encoding
    onehot = F.one_hot(mask.long(), num_classes=onehot_classes) # Shape: [B, D, H, W, K]
    onehot = onehot.permute(0, 4, 1, 2, 3).contiguous() # Shape: [B, K, D, H, W]

    # Exclude background channel (index 0)
    P = onehot[:, 1:, :, :, :].float() # Shape: [B, C, D, H, W] where C = K-1
    
    # Important: Match the output size to what soft_adjacency produces
    # soft_adjacency uses prob[:, 1:] which gives C = num_classes-1
    # So we need to return adjacency of size [B, num_classes-1, num_classes-1]
    C_target = num_classes - 1  # This is what we want
    C_actual = P.shape[1]        # This is what we have
    
    if C_actual != C_target:
        # Pad or truncate to match
        if C_actual < C_target:
            # Pad with zeros
            padding = torch.zeros(
                P.shape[0], C_target - C_actual, *P.shape[2:],
                device=P.device, dtype=P.dtype
            )
            P = torch.cat([P, padding], dim=1)
        else:
            # Truncate (shouldn't happen but handle it)
            P = P[:, :C_target]

    # Get neighborhood offsets
    offsets = connectivity_offsets(connectivity)

    B = P.shape[0]
    C = P.shape[1]
    A = mask.new_zeros((B, C, C), dtype=torch.float32)

    # Compute adjacency
    for (dx, dy, dz) in offsets:
        P_shift = shift3d_pad(P, dx, dy, dz)
        interaction = torch.einsum("bcxyz, bkxyz -> bck", P, P_shift)
        A = A + interaction

    # Make symmetric and remove diagonal
    A = 0.5 * (A + A.transpose(1, 2))
    A = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
    
    return A


