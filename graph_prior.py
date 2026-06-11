from typing import List, Tuple, Literal
import torch


def connectivity_offsets(connectivity: Literal['6-connectivity', '18-connectivity', '26-connectivity']) -> List[Tuple[int, int, int]]:
    assert connectivity in ['6-connectivity', '18-connectivity', '26-connectivity'], f"connectivity must be one of '6-connectivity', '18-connectivity', '26-connectivity', got {connectivity}"

    offs: List[Tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue

                manhattan_dist = abs(dx) + abs(dy) + abs(dz)

                if connectivity == '6-connectivity' and manhattan_dist == 1:
                    offs.append((dx, dy, dz))
                elif connectivity == '18-connectivity' and (manhattan_dist == 1 or manhattan_dist == 2 and max(abs(dx), abs(dy), abs(dz)) == 1):
                    offs.append((dx, dy, dz))
                elif connectivity == '26-connectivity':
                    offs.append((dx, dy, dz))
    
    return offs


def shift3d_pad(tensor: torch.Tensor, dx: int, dy: int, dz: int) -> torch.Tensor:
    """
    Shift a 5D tensor [B, C, D, H, W] by (dx, dy, dz) with zero padding.
    
    This function enables differentiable spatial shifting for adjacency computation.
    Unlike circular shifts, this uses zero padding to handle boundaries properly.
    
    Args:
        tensor: Input tensor [B, C, D, H, W] 
        dx: Shift amount in D dimension (-D < dx < D)
        dy: Shift amount in H dimension (-H < dy < H)  
        dz: Shift amount in W dimension (-W < dz < W)
        
    Returns:
        Shifted tensor with same shape, zero-padded at boundaries
        
    Examples:
        >>> x = torch.randn(2, 3, 10, 10, 10)
        >>> shifted = shift3d_pad(x, 1, 0, -1)  # Move +1 in D, -1 in W
        >>> shifted.shape
        torch.Size([2, 3, 10, 10, 10])
    """
    assert tensor.ndim == 5, f"Expected 5D tensor [B,C,D,H,W], got {tensor.ndim}D"
    
    B, C, D, H, W = tensor.shape
    
    # Calculate valid regions after shift
    d_start, d_end = max(0, dx), D + min(0, dx)
    h_start, h_end = max(0, dy), H + min(0, dy)  
    w_start, w_end = max(0, dz), W + min(0, dz)
    
    # Calculate source regions (inverse shift)
    d_src_start, d_src_end = d_start - dx, d_end - dx
    h_src_start, h_src_end = h_start - dy, h_end - dy
    w_src_start, w_src_end = w_start - dz, w_end - dz
    
    # Create output tensor (initialized to zeros)
    output = torch.zeros_like(tensor)
    
    # Copy valid region
    if d_end > d_start and h_end > h_start and w_end > w_start:
        output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = \
            tensor[:, :, d_src_start:d_src_end, h_src_start:h_src_end, w_src_start:w_src_end]
    
    return output


def validate_tensor_shapes(prob: torch.Tensor, label: torch.Tensor = None) -> None:
    """
    Validate tensor shapes for graph computation.
    
    Args:
        prob: [B, K, D, H, W] probability tensor
        label: Optional [B, D, H, W] label tensor
        
    Raises:
        ValueError: If shapes are invalid
        
    Examples:
        >>> prob = torch.randn(2, 5, 32, 32, 32)
        >>> validate_tensor_shapes(prob)  # Should pass
        >>> validate_tensor_shapes(prob, torch.randint(0, 5, (2, 32, 32, 32)))  # Should pass
    """
    # Validate prob tensor
    if prob.ndim != 5:
        raise ValueError(f"prob tensor must be 5D [B,K,D,H,W], got {prob.ndim}D with shape {prob.shape}")
    
    B, K, D, H, W = prob.shape
    
    if K < 2:
        raise ValueError(f"Number of classes K must be >= 2, got {K}")
    
    # Validate label tensor if provided
    if label is not None:
        if label.ndim != 4:
            raise ValueError(f"label tensor must be 4D [B,D,H,W], got {label.ndim}D with shape {label.shape}")
        
        if label.shape != (B, D, H, W):
            raise ValueError(f"Shape mismatch: prob {prob.shape} vs label {label.shape}")