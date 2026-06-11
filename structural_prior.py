"""
FastSurfer 78-class를 위한 정확한 anatomical priors.
"""
import os
import torch
from typing import Tuple, Dict, List, Optional

_DEFAULT_ATLAS_PRIOR_PATH = os.environ.get(
    "ATLAS_PRIOR_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas_prior_78class.pt")
)

def load_atlas_prior(atlas_path: str = None) -> Optional[dict]:
    """
    atlas_prior_78class.pt 파일 로드 시도.
    
    Returns:
        prior dict if found, None otherwise
    """
    path = atlas_path or _DEFAULT_ATLAS_PRIOR_PATH
    
    if not os.path.exists(path):
        return None
    
    try:
        import numpy as np
        prior = torch.load(path, map_location='cpu', weights_only=False)
        
        # numpy → torch 변환 (혹시 numpy로 저장된 경우)
        for key in ["adjacency_binary", "adjacency_strength", "possible_mask", "raw_contact_counts"]:
            if key in prior and isinstance(prior[key], np.ndarray):
                prior[key] = torch.from_numpy(prior[key])
        
        return prior
    except Exception as e:
        print(f"⚠ Atlas prior 로드 실패 ({path}): {e}")
        return None

def get_atlas_based_79class_prior(atlas_path: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Atlas에서 추출한 data-driven prior 반환.
    
    build_atlas_prior.py로 생성한 atlas_prior_78class.pt에서:
      - possible_mask: [78, 78] 실제 atlas 인접 여부
      - expected_strength: [78, 78] 정규화된 접촉 강도
    
    Returns:
        possible_mask: [78, 78] tensor (1=인접 가능, 0=불가능)
        expected_strength: [78, 78] tensor (0~1, 인접 강도)
    """
    prior = load_atlas_prior(atlas_path)
    
    if prior is None:
        raise FileNotFoundError(
            f"Atlas prior 파일을 찾을 수 없습니다: {atlas_path or _DEFAULT_ATLAS_PRIOR_PATH}\n"
            f"build_atlas_prior.py를 먼저 실행하세요:\n"
            f"  python build_atlas_prior.py --atlas /path/to/aparc+aseg.mgz --output atlas_prior_78class.pt"
        )
    
    possible_mask = prior["possible_mask"].float()
    expected_strength = prior["adjacency_strength"].float()
    
    # 검증
    target_dim = 78
    assert possible_mask.shape == (target_dim, target_dim), f"Expected [{target_dim},{target_dim}], got {possible_mask.shape}"
    assert expected_strength.shape == (target_dim, target_dim), f"Expected [{target_dim},{target_dim}], got {expected_strength.shape}"
    
    # 대칭 보장
    possible_mask = 0.5 * (possible_mask + possible_mask.T)
    expected_strength = 0.5 * (expected_strength + expected_strength.T)
    
    # 대각선 제거
    possible_mask = possible_mask - torch.diag(torch.diag(possible_mask))
    expected_strength = expected_strength - torch.diag(torch.diag(expected_strength))
    
    # strength는 possible한 곳에서만
    expected_strength = expected_strength * (possible_mask > 0).float()
    
    print(f"✓ Atlas-based prior loaded: {int(possible_mask.sum().item())//2} edges, "
          f"density={possible_mask.sum().item()/(target_dim*(target_dim-1)):.3f}")
    
    return possible_mask, expected_strength

# ================================================================
# 기존 수동 prior (fallback용으로 유지)
# ================================================================

def get_5class_hard_prior() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    5-class simplified segmentation의 Hard Prior.
    
    Classes (0-indexed in tensors, but conceptually):
        0: Background (not in tensor)
        1: Cortical Gray Matter  -> tensor index 0
        2: Subcortical Gray Matter -> tensor index 1
        3: White Matter -> tensor index 2
        4: CSF -> tensor index 3
    
    Returns:
        possible_mask: [4, 4] 인접 가능하면 1, 불가능하면 0
        expected_strength: [4, 4] 기대되는 인접 강도 (0~1)
    """
    C = 4  # Foreground classes only
    
    # 5-class는 큰 영역이므로 모든 인접이 가능
    possible_mask = torch.ones(C, C)
    possible_mask = possible_mask - torch.eye(C)  # 자기 자신 제외
    
    # 기대 강도 (얼마나 자주/강하게 인접하는지)
    expected_strength = torch.zeros(C, C)
    
    # Cortical GM (idx 0) ↔ others
    expected_strength[0, 1] = 0.3   # ↔ Subcortical (가끔)
    expected_strength[0, 2] = 1.0   # ↔ WM (항상, 강하게)
    expected_strength[0, 3] = 0.7   # ↔ CSF (sulci, 자주)
    
    # Subcortical GM (idx 1) ↔ others
    expected_strength[1, 2] = 1.0   # ↔ WM (항상)
    expected_strength[1, 3] = 0.5   # ↔ CSF (ventricle 경계)
    
    # WM (idx 2) ↔ CSF (idx 3)
    expected_strength[2, 3] = 0.8   # ventricle 경계
    
    # Symmetric
    expected_strength = expected_strength + expected_strength.T
    
    return possible_mask, expected_strength

def get_79class_hard_prior_manual() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    79-class FastSurfer의 Hard Prior.
    
    FastSurfer labels (1-indexed in data):
    - 1-33: Subcortical structures
    - 34-64: Left hemisphere cortical
    - 65-78: Right hemisphere cortical
    
    핵심 규칙:
    - 좌/우 반구 구조는 직접 인접 불가능 (midline 제외)
    - 같은 반구 내에서만 인접 가능
    - Midline 구조는 양쪽과 가능
    
    Returns:
        possible_mask: [78, 78] 인접 가능 여부 (78 = 78 classes - 1 background)
        expected_strength: [78, 78] 기대 강도
    """
    C = 78  # 78 classes - 1 background
    
    # 기본: 같은 반구만 가능
    possible_mask = torch.zeros(C, C)
    expected_strength = torch.zeros(C, C)
    
    # ============================================================
    # Hemisphere definitions (0-indexed for tensor access)
    # ============================================================
    # Left hemisphere structures
    left_wm = [0]  # Cortical-white-matter-lh (label 1)
    left_ventricles = [1, 2]  # Lateral, Inferior Lateral (labels 2, 3)
    left_cerebellum = [3, 4]  # WM and Cortex (labels 4, 5)
    left_deep_gray = [5, 6, 7, 8, 12, 13, 15, 16]  # Thalamus, Caudate, etc (labels 6-9, 13-14, 16-17)
    left_choroid = [17]  # (label 18)
    left_cortical = list(range(33, 64))  # Labels 34-64 (0-indexed: 33-63)
    
    # Right hemisphere structures
    right_wm = [18]  # Cortical-white-matter-rh (label 19)
    right_ventricles = [19, 20]  # Lateral, Inferior Lateral (labels 20, 21)
    right_cerebellum = [21, 22]  # WM and Cortex (labels 22, 23)
    right_deep_gray = [23, 24, 25, 26, 27, 28, 29, 30]  # (labels 24-31)
    right_choroid = [31]  # (label 32)
    right_cortical = list(range(64, 78))  # Labels 65-78 (0-indexed: 64-77)
    
    # Midline structures (can connect to both hemispheres)
    midline = [9, 10, 11, 14, 32]  # 3rd-V, 4th-V, Brain-Stem, CSF, WM-hypo
    # Labels: 10, 11, 12, 15, 33 (0-indexed: 9, 10, 11, 14, 32)
    
    # ============================================================
    # Same hemisphere: always possible
    # ============================================================
    left_all = (left_wm + left_ventricles + left_cerebellum + 
                left_deep_gray + left_choroid + left_cortical)
    right_all = (right_wm + right_ventricles + right_cerebellum + 
                 right_deep_gray + right_choroid + right_cortical)
    
    for hemisphere_structures in [left_all, right_all]:
        for i in hemisphere_structures:
            for j in hemisphere_structures:
                if i != j:
                    possible_mask[i, j] = 1
                    expected_strength[i, j] = 0.2  # Default moderate
    
    # ============================================================
    # Midline connections: 양쪽 모두와 가능
    # ============================================================
    all_structures = left_all + right_all
    
    for m in midline:
        for s in all_structures:
            possible_mask[m, s] = 1
            possible_mask[s, m] = 1
            expected_strength[m, s] = 0.3
            expected_strength[s, m] = 0.3
    
    # Midline끼리도 가능
    for m1 in midline:
        for m2 in midline:
            if m1 != m2:
                possible_mask[m1, m2] = 1
                expected_strength[m1, m2] = 0.5
    
    # ============================================================
    # White matter connections (STRONG)
    # ============================================================
    # Left WM <-> Left cortical (very strong, always adjacent)
    for c in left_cortical:
        possible_mask[left_wm[0], c] = 1
        possible_mask[c, left_wm[0]] = 1
        expected_strength[left_wm[0], c] = 1.0
        expected_strength[c, left_wm[0]] = 1.0
    
    # Right WM <-> Right cortical
    for c in right_cortical:
        possible_mask[right_wm[0], c] = 1
        possible_mask[c, right_wm[0]] = 1
        expected_strength[right_wm[0], c] = 1.0
        expected_strength[c, right_wm[0]] = 1.0
    
    # WM <-> Deep gray matter
    for wm_idx, deep_gray_list in [(left_wm[0], left_deep_gray), 
                                     (right_wm[0], right_deep_gray)]:
        for dg in deep_gray_list:
            possible_mask[wm_idx, dg] = 1
            possible_mask[dg, wm_idx] = 1
            expected_strength[wm_idx, dg] = 0.8
            expected_strength[dg, wm_idx] = 0.8
    
    # ============================================================
    # Ventricle connections (moderate-strong)
    # ============================================================
    # Ventricles <-> Deep gray matter (ventricular boundaries)
    for vent_list, deep_gray_list in [(left_ventricles, left_deep_gray),
                                        (right_ventricles, right_deep_gray)]:
        for v in vent_list:
            for dg in deep_gray_list:
                possible_mask[v, dg] = 1
                possible_mask[dg, v] = 1
                expected_strength[v, dg] = 0.7
                expected_strength[dg, v] = 0.7
    
    # ============================================================
    # Cerebellar connections
    # ============================================================
    # Cerebellum WM <-> Cerebellum Cortex
    for cereb_wm, cereb_ctx in [(left_cerebellum[0], left_cerebellum[1]),
                                  (right_cerebellum[0], right_cerebellum[1])]:
        possible_mask[cereb_wm, cereb_ctx] = 1
        possible_mask[cereb_ctx, cereb_wm] = 1
        expected_strength[cereb_wm, cereb_ctx] = 1.0
        expected_strength[cereb_ctx, cereb_wm] = 1.0
    
    # Cerebellum <-> Brainstem
    brainstem_idx = 11  # 0-indexed for label 12
    for cereb_list in [left_cerebellum, right_cerebellum]:
        for c in cereb_list:
            possible_mask[c, brainstem_idx] = 1
            possible_mask[brainstem_idx, c] = 1
            expected_strength[c, brainstem_idx] = 0.9
            expected_strength[brainstem_idx, c] = 0.9
    
    # ============================================================
    # Deep gray matter internal connections (boost strength)
    # ============================================================
    for deep_gray_list in [left_deep_gray, right_deep_gray]:
        for i in deep_gray_list:
            for j in deep_gray_list:
                if i != j:
                    expected_strength[i, j] = 0.6
    
    # ============================================================
    # CSF connections
    # ============================================================
    # CSF (label 15, idx 14) <-> Ventricles
    csf_idx = 14
    all_ventricles = left_ventricles + right_ventricles + [9, 10]  # Include 3rd, 4th
    for v in all_ventricles:
        possible_mask[csf_idx, v] = 1
        possible_mask[v, csf_idx] = 1
        expected_strength[csf_idx, v] = 0.8
        expected_strength[v, csf_idx] = 0.8

    # External CSF (label 79, idx 78) <-> Cortical ### 이 부분 지울수도 있음
    # ext_csf_idx = 78
    # for c in left_cortical + right_cortical:
    #     possible_mask[ext_csf_idx, c] = 1
    #     possible_mask[c, ext_csf_idx] = 1
    #     expected_strength[ext_csf_idx, c] = 0.6
    #     expected_strength[c, ext_csf_idx] = 0.6
    
    # ============================================================
    # Ensure symmetry
    # ============================================================
    possible_mask = 0.5 * (possible_mask + possible_mask.T)
    expected_strength = 0.5 * (expected_strength + expected_strength.T)
    
    # Remove diagonal
    possible_mask = possible_mask - torch.diag(torch.diag(possible_mask))
    expected_strength = expected_strength - torch.diag(torch.diag(expected_strength))
    
    # Ensure expected_strength only where possible
    expected_strength = expected_strength * possible_mask
    
    return possible_mask, expected_strength

def get_priors_for_num_classes(num_classes: int, atlas_path: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prior 반환. 79-class의 경우 atlas prior를 우선 사용하고,
    파일이 없으면 수동 prior로 fallback.
    
    Args:
        num_classes: Total number of classes including background
        atlas_path: Optional path to atlas_prior_79class.pt
        
    Returns:
        possible_mask: [C, C] where C = num_classes - 1
        expected_strength: [C, C]
    """
    if num_classes == 5:
        return get_5class_hard_prior()
    
    elif num_classes == 79:
        # 1순위: atlas-based data-driven prior
        try:
            return get_atlas_based_79class_prior(atlas_path)
        except FileNotFoundError:
            # 2순위: 수동 하드코딩 prior (fallback)
            print("⚠ Atlas prior 파일 없음 → 수동 prior 사용 (fallback)")
            print("  더 정확한 prior를 위해 build_atlas_prior.py를 실행하세요.")
            return get_79class_hard_prior_manual()
    
    else:
        C = num_classes - 1
        possible_mask = torch.ones(C, C) - torch.eye(C)
        expected_strength = torch.ones(C, C) * 0.5 - torch.eye(C) * 0.5
        print(f"⚠ No specific prior for {num_classes} classes, using default")
        return possible_mask, expected_strength


def get_adjacency_from_atlas(atlas_path: str) -> torch.Tensor:
    """
    Load actual adjacency from FreeSurfer/FastSurfer atlas.
    → build_atlas_prior.py로 대체됨. 호환성을 위해 유지.
    """
    prior = load_atlas_prior(atlas_path)
    if prior is None:
        raise FileNotFoundError(f"Atlas prior not found: {atlas_path}")
    return prior["adjacency_binary"]


def validate_prior(
    possible_mask: torch.Tensor,
    expected_strength: torch.Tensor,
) -> dict:
    """
    Validate structural prior consistency.
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check symmetry
    if not torch.allclose(possible_mask, possible_mask.T, atol=1e-6):
        issues.append("possible_mask is not symmetric")
    
    if not torch.allclose(expected_strength, expected_strength.T, atol=1e-6):
        issues.append("expected_strength is not symmetric")
    
    # Check diagonal
    if possible_mask.diag().sum() > 0:
        issues.append(f"possible_mask has non-zero diagonal")
    
    if expected_strength.diag().sum() > 0:
        issues.append(f"expected_strength has non-zero diagonal")
    
    # Check consistency
    inconsistent = (expected_strength > 0) & (possible_mask == 0)
    if inconsistent.sum() > 0:
        issues.append(f"{inconsistent.sum()} entries have strength but marked impossible")
    
    # Statistics
    num_possible = (possible_mask > 0).sum().item() // 2
    C = possible_mask.shape[0]
    total_possible = C * (C - 1) // 2
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "num_possible_edges": num_possible,
        "total_possible_edges": total_possible,
        "density": num_possible / total_possible if total_possible > 0 else 0,
        "sparsity": (possible_mask == 0).float().mean().item(),
    }