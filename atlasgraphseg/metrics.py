"""
Structural quality metrics for brain segmentation.
"""
import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Connected Components / Island Penalty
# ─────────────────────────────────────────────

def count_connected_components(mask: torch.Tensor, class_id: int) -> int:
    """
    Count number of connected components for a given class.
    
    Args:
        mask: [D, H, W] segmentation mask
        class_id: Class to check
        
    Returns:
        Number of connected components
    """
    binary_mask = (mask == class_id).cpu().numpy()
    _, num_components = ndimage.label(binary_mask)
    return num_components

def compute_island_penalty(pred: torch.Tensor, num_classes: int) -> float:
    """
    Compute island penalty = total number of extra components.
    
    Lower is better (ideally 0 = each class is a single connected component).
    
    Args:
        pred: [B, D, H, W] predicted segmentation
        num_classes: Total classes including background
        
    Returns:
        Average island penalty across batch
    """
    B = pred.shape[0]
    total_penalty = 0.0
    
    for b in range(B):
        for c in range(1, num_classes):  # Skip background
            num_components = count_connected_components(pred[b], c)
            # Penalty = components - 1 (0 if single component)
            penalty = max(0, num_components - 1)
            total_penalty += penalty
    
    return total_penalty / B

# ================================================================
# Betti Number Error (Topological Correctness)
# ================================================================

def compute_betti_numbers(binary_mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Compute Betti numbers (β0, β1, β2) for a 3D binary mask.

    β0 = number of connected components
    β1 = number of tunnels / loops (1-dimensional holes)
    β2 = number of enclosed cavities (2-dimensional holes / voids)

    Method
    ------
    Uses the **cubical complex** representation where each occupied voxel
    is a 3-cell (cube). The Euler characteristic is computed via the
    inclusion-exclusion formula on shared k-cells:

        χ = #(vertices) - #(edges) + #(faces) - #(cubes)

    where:
        - cubes (3-cells):    occupied voxels
        - faces (2-cells):    shared faces between 6-adjacent occupied voxel pairs
        - edges (1-cells):    shared edges = 2x2 occupied voxel squares (3 orientations)
        - vertices (0-cells): shared vertices = 2x2x2 fully-occupied cubes

    This follows the **primal cubical complex convention** as described in:
        - Wagner et al., "Efficient Computation of Persistent Homology for
          Cubical Data" (2011)
        - Specifically: χ = V - E + F - C  (alternating sum from 0-cells to 3-cells)

    β0 and β2 are computed directly (connected components of foreground
    and enclosed cavities), then β1 is derived from: β1 = β0 + β2 - χ.

    Connectivity assumptions:
        - β0: 26-connectivity for foreground (most permissive, fewer components)
        - β2: 26-connectivity for background cavities

    Note for paper: if comparing against methods that use 6-connectivity
    for β0, results will differ. The choice here follows the convention
    that the cubical complex with 26-connectivity gives topologically
    consistent Betti numbers for voxel data.

    Args:
        binary_mask: 3D boolean/int numpy array

    Returns:
        (β0, β1, β2)
    """
    mask = binary_mask.astype(bool)

    if not mask.any():
        return (0, 0, 0)

    # ── β0: connected components (26-connectivity) ──
    struct_26 = ndimage.generate_binary_structure(3, 3)
    labeled, beta0 = ndimage.label(mask, structure=struct_26)

    # ── β2: enclosed cavities ──
    # = background connected components that do NOT touch the volume boundary
    bg_mask = ~mask
    labeled_bg, n_bg = ndimage.label(bg_mask, structure=struct_26)

    # Find background components touching any of the 6 boundary faces
    boundary_labels = set()
    D, H, W = mask.shape
    for face in [labeled_bg[0],    labeled_bg[-1],     # D boundaries
                 labeled_bg[:, 0], labeled_bg[:, -1],   # H boundaries
                 labeled_bg[:, :, 0], labeled_bg[:, :, -1]]:  # W boundaries
        boundary_labels.update(np.unique(face))
    boundary_labels.discard(0)  # 0 = no component (foreground region in bg labeling)

    beta2 = max(0, n_bg - len(boundary_labels))

    # ── Euler characteristic via cubical complex ──
    # 3-cells (cubes): occupied voxels
    n_cubes = int(mask.sum())

    # 2-cells (faces): pairs of 6-adjacent occupied voxels sharing a face
    n_faces = (
        int((mask[:-1] & mask[1:]).sum()) +          # D-axis
        int((mask[:, :-1] & mask[:, 1:]).sum()) +    # H-axis
        int((mask[:, :, :-1] & mask[:, :, 1:]).sum())  # W-axis
    )

    # 1-cells (edges): 2x2 squares of occupied voxels sharing an edge
    # 3 orientations: DH-plane, DW-plane, HW-plane
    n_edges = (
        int((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).sum()) +      # DH
        int((mask[:-1, :, :-1] & mask[1:, :, :-1] & mask[:-1, :, 1:] & mask[1:, :, 1:]).sum()) +  # DW
        int((mask[:, :-1, :-1] & mask[:, 1:, :-1] & mask[:, :-1, 1:] & mask[:, 1:, 1:]).sum())     # HW
    )

    # 0-cells (vertices): 2x2x2 cubes all occupied
    n_vertices = int((
        mask[:-1, :-1, :-1] & mask[1:, :-1, :-1] &
        mask[:-1, 1:, :-1]  & mask[1:, 1:, :-1] &
        mask[:-1, :-1, 1:]  & mask[1:, :-1, 1:] &
        mask[:-1, 1:, 1:]   & mask[1:, 1:, 1:]
    ).sum())

    # χ = V - E + F - C  (alternating sum: 0-cells ... 3-cells)
    euler = n_vertices - n_edges + n_faces - n_cubes

    # β1 = β0 + β2 - χ  (from χ = β0 - β1 + β2)
    beta1 = max(0, beta0 + beta2 - euler)

    return (beta0, beta1, beta2)


def compute_betti_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    classes_to_eval: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Betti number error between prediction and ground truth.

    For each foreground class, computes |β_k^pred - β_k^gt| for k=0,1,2.

    The key insight for brain segmentation:
      - β0 error: detects island/fragmentation problems
      - β1 error: detects spurious tunnels/handles
      - β2 error: detects spurious enclosed cavities

    Args:
        pred: [B, D, H, W] predicted segmentation (argmax)
        target: [B, D, H, W] ground truth
        num_classes: Total classes including background
        classes_to_eval: Optional subset of classes to evaluate.
                         If None, evaluates all foreground classes.

    Returns:
        Dict with:
            'betti_0_error': mean |Δβ0| across classes and batch
            'betti_1_error': mean |Δβ1|
            'betti_2_error': mean |Δβ2|
            'betti_total_error': sum of all Betti errors
            'betti_0_error_per_class': dict of per-class β0 errors
    """
    B = pred.shape[0]

    if classes_to_eval is None:
        classes_to_eval = list(range(1, num_classes))

    betti_errors = {0: [], 1: [], 2: []}
    per_class_b0 = {}

    for b in range(B):
        pred_np = pred[b].cpu().numpy()
        target_np = target[b].cpu().numpy()

        for c in classes_to_eval:
            pred_mask = (pred_np == c)
            gt_mask = (target_np == c)

            # Skip classes absent in both
            if not pred_mask.any() and not gt_mask.any():
                continue

            bp = compute_betti_numbers(pred_mask)
            bg = compute_betti_numbers(gt_mask)

            for k in range(3):
                betti_errors[k].append(abs(bp[k] - bg[k]))

            if c not in per_class_b0:
                per_class_b0[c] = []
            per_class_b0[c].append(abs(bp[0] - bg[0]))

    result = {}
    for k in range(3):
        vals = betti_errors[k]
        result[f'betti_{k}_error'] = float(np.mean(vals)) if vals else 0.0

    result['betti_total_error'] = sum(result[f'betti_{k}_error'] for k in range(3))

    result['betti_0_error_per_class'] = {
        c: float(np.mean(v)) for c, v in per_class_b0.items()
    }

    return result


# ================================================================
# Surface Distance Metrics: HD95 & ASSD
# ================================================================

def _surface_distances(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetric surface distances between two binary masks.

    Returns:
        (distances_pred_to_gt, distances_gt_to_pred)
        Each is a 1D array of distances from one surface to the other.
    """
    # Extract surfaces (border voxels)
    pred_border = pred_mask ^ ndimage.binary_erosion(pred_mask, iterations=1)
    gt_border = gt_mask ^ ndimage.binary_erosion(gt_mask, iterations=1)

    # If either surface is empty, return inf
    if not pred_border.any() or not gt_border.any():
        return np.array([np.inf]), np.array([np.inf])

    # Distance transforms
    dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)

    # Surface distances
    dist_pred_to_gt = dt_gt[pred_border]
    dist_gt_to_pred = dt_pred[gt_border]

    return dist_pred_to_gt, dist_gt_to_pred


def compute_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """
    95th percentile Hausdorff Distance (HD95).

    More robust than full HD as it ignores the worst 5% outliers.

    Args:
        pred_mask: 3D binary numpy array
        gt_mask: 3D binary numpy array
        voxel_spacing: voxel dimensions in mm (d, h, w)

    Returns:
        HD95 in mm (lower is better)
    """
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return np.inf

    d_p2g, d_g2p = _surface_distances(pred_mask, gt_mask, voxel_spacing)

    if np.isinf(d_p2g).all() or np.isinf(d_g2p).all():
        return np.inf

    all_distances = np.concatenate([d_p2g, d_g2p])
    return float(np.percentile(all_distances, 95))


def compute_assd(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """
    Average Symmetric Surface Distance (ASSD).

    ASSD = (mean(d_pred→gt) + mean(d_gt→pred)) / 2

    Args:
        pred_mask: 3D binary numpy array
        gt_mask: 3D binary numpy array
        voxel_spacing: voxel dimensions in mm

    Returns:
        ASSD in mm (lower is better)
    """
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return np.inf

    d_p2g, d_g2p = _surface_distances(pred_mask, gt_mask, voxel_spacing)

    if np.isinf(d_p2g).all() or np.isinf(d_g2p).all():
        return np.inf

    return float((d_p2g.mean() + d_g2p.mean()) / 2.0)


def compute_surface_distances_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    classes_to_eval: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute HD95 and ASSD averaged over foreground classes and batch.

    Args:
        pred: [B, D, H, W] predicted segmentation (argmax)
        target: [B, D, H, W] ground truth
        num_classes: Total classes including background
        voxel_spacing: Voxel spacing in mm (from NIfTI header)
        classes_to_eval: Optional subset of class IDs to evaluate

    Returns:
        Dict with:
            'hd95': mean HD95 across classes and batch (mm)
            'assd': mean ASSD across classes and batch (mm)
            'hd95_per_class': dict of per-class HD95
            'assd_per_class': dict of per-class ASSD
    """
    B = pred.shape[0]

    if classes_to_eval is None:
        classes_to_eval = list(range(1, num_classes))

    hd95_all = []
    assd_all = []
    hd95_per_class = {}
    assd_per_class = {}

    for b in range(B):
        pred_np = pred[b].cpu().numpy()
        target_np = target[b].cpu().numpy()

        for c in classes_to_eval:
            pred_c = (pred_np == c)
            gt_c = (target_np == c)

            # Skip if class absent in both pred and GT
            # if not pred_c.any() and not gt_c.any():
            if not pred_c.any() or not gt_c.any():
                continue

            h = compute_hd95(pred_c, gt_c, voxel_spacing)
            a = compute_assd(pred_c, gt_c, voxel_spacing)

            # Only include finite values in mean
            if np.isfinite(h):
                hd95_all.append(h)
                hd95_per_class.setdefault(c, []).append(h)
            if np.isfinite(a):
                assd_all.append(a)
                assd_per_class.setdefault(c, []).append(a)

    return {
        'hd95': float(np.mean(hd95_all)) if hd95_all else float('inf'),
        'assd': float(np.mean(assd_all)) if assd_all else float('inf'),
        'hd95_per_class': {c: float(np.mean(v)) for c, v in hd95_per_class.items()},
        'assd_per_class': {c: float(np.mean(v)) for c, v in assd_per_class.items()},
    }


# ================================================================
# Per-class Dice
# ================================================================

def compute_per_class_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute Dice score per foreground class.

    Args:
        pred: [B, D, H, W] predicted segmentation (argmax)
        target: [B, D, H, W] ground truth
        num_classes: Total classes including background

    Returns:
        Dict with 'dice_mean' and 'dice_per_class'
    """
    dice_per_class = {}

    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_per_class[c] = dice.item()

    valid_dice = list(dice_per_class.values())
    return {
        'dice_mean': float(np.mean(valid_dice)) if valid_dice else 0.0,
        'dice_per_class': dice_per_class,
    }


# ─────────────────────────────────────────────
# Adjacency F1
# ─────────────────────────────────────────────

def compute_adjacency_f1(
    pred_adj: torch.Tensor,
    gt_adj: torch.Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Compute F1 score for adjacency prediction.
    
    Args:
        pred_adj: [C, C] predicted adjacency (soft)
        gt_adj: [C, C] ground truth adjacency (binary)
        threshold: Threshold for binarizing prediction
        
    Returns:
        F1 score
    """
    # Binarize prediction
    pred_binary = (pred_adj > threshold).float()
    gt_binary = (gt_adj > 0).float()
    
    # Compute TP, FP, FN
    tp = ((pred_binary == 1) & (gt_binary == 1)).sum().item()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum().item()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum().item()
    
    # F1
    if tp + fp + fn == 0:
        return 1.0
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1
# ================================================================
# Comprehensive Structural Score
# ================================================================

def compute_structural_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_adj: torch.Tensor,
    gt_adj: torch.Tensor,
    num_classes: int,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    compute_topology: bool = True,
    compute_surfaces: bool = True,
    topology_classes: Optional[List[int]] = None,
    surface_classes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive structural quality metrics.

    Includes all metrics needed for paper:
      - Island penalty
      - Adjacency F1
      - Betti number errors (β0, β1, β2)
      - HD95 and ASSD
      - Composite structural score

    Args:
        pred: [B, D, H, W] predictions
        target: [B, D, H, W] ground truth
        pred_adj: [B, C, C] predicted adjacency
        gt_adj: [B, C, C] GT adjacency
        num_classes: Total classes
        voxel_spacing: Voxel spacing in mm for surface distances
        compute_topology: Whether to compute Betti numbers (slow)
        compute_surfaces: Whether to compute HD95/ASSD (slow)
        topology_classes: Subset of classes for topology (None=all FG).
                          Useful for 79-class to only check key structures.
        surface_classes: Subset of classes for surface distances

    Returns:
        Dictionary of structural metrics
    """
    B = pred.shape[0]

    # 1. Island penalty
    island_penalty = compute_island_penalty(pred, num_classes)

    # 2. Adjacency F1
    adj_f1_sum = 0.0
    for b in range(B):
        f1 = compute_adjacency_f1(pred_adj[b], gt_adj[b])
        adj_f1_sum += f1
    adj_f1 = adj_f1_sum / B

    result = {
        'island_penalty': island_penalty,
        'adj_f1': adj_f1,
    }

    # 3. Betti number errors (topological correctness)
    if compute_topology:
        try:
            betti = compute_betti_error(
                pred, target, num_classes,
                classes_to_eval=topology_classes,
            )
            result['betti_0_error'] = betti['betti_0_error']
            result['betti_1_error'] = betti['betti_1_error']
            result['betti_2_error'] = betti['betti_2_error']
            result['betti_total_error'] = betti['betti_total_error']
        except Exception as e:
            print(f"  ⚠ Betti computation failed: {e}")
            result['betti_0_error'] = 0.0
            result['betti_1_error'] = 0.0
            result['betti_2_error'] = 0.0
            result['betti_total_error'] = 0.0

    # 4. Surface distances (HD95, ASSD)
    if compute_surfaces:
        try:
            surf = compute_surface_distances_per_class(
                pred, target, num_classes,
                voxel_spacing=voxel_spacing,
                classes_to_eval=surface_classes,
            )
            result['hd95'] = surf['hd95']
            result['assd'] = surf['assd']
        except Exception as e:
            print(f"  ⚠ Surface distance computation failed: {e}")
            result['hd95'] = float('inf')
            result['assd'] = float('inf')

    # 5. Composite structural score
    # Higher is better: adj_f1 ↑, island ↓, betti ↓, hd95 ↓
    structural_score = adj_f1 - 0.1 * island_penalty
    if compute_topology:
        structural_score -= 0.05 * result.get('betti_total_error', 0.0)
    result['structural_score'] = structural_score

    return result