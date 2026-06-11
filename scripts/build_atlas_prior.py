"""
build_atlas_prior.py
====================
FreeSurfer fsaverage aparc+aseg.mgz → FastSurfer 79-class Atlas Graph Prior

핵심:
  1) aparc+aseg.mgz의 FreeSurfer 라벨(2,3,...,2035)을
     FastSurfer 79-class 라벨(1-79, 0=bg)로 remap
  2) remap된 볼륨에서 voxel-level adjacency 계산
  3) [79, 79] possible_mask, adjacency_strength 산출

Usage:
  python scripts/build_atlas_prior.py \
      --atlas assets/aparc+aseg.mgz \
      --output assets/priors/atlas_prior_78class.pt --connectivity 6 --visualize
"""
import argparse, sys
from pathlib import Path
from collections import OrderedDict
import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("pip install nibabel --break-system-packages")


# ================================================================
# FreeSurfer label → FastSurfer 78-class (1-78) 매핑
# ================================================================
FREESURFER_TO_FASTSURFER = {
    # --- Subcortical ---
    2:   1,   # Left-Cerebral-White-Matter
    4:   2,   # Left-Lateral-Ventricle
    5:   3,   # Left-Inf-Lat-Vent
    7:   4,   # Left-Cerebellum-White-Matter
    8:   5,   # Left-Cerebellum-Cortex
    10:  6,   # Left-Thalamus
    11:  7,   # Left-Caudate
    12:  8,   # Left-Putamen
    13:  9,   # Left-Pallidum
    14:  10,  # 3rd-Ventricle
    15:  11,  # 4th-Ventricle
    16:  12,  # Brain-Stem
    17:  13,  # Left-Hippocampus
    18:  14,  # Left-Amygdala
    24:  15,  # CSF
    26:  16,  # Left-Accumbens-area
    28:  17,  # Left-VentralDC
    31:  18,  # Left-choroid-plexus
    41:  19,  # Right-Cerebral-White-Matter
    43:  20,  # Right-Lateral-Ventricle
    44:  21,  # Right-Inf-Lat-Vent
    46:  22,  # Right-Cerebellum-White-Matter
    47:  23,  # Right-Cerebellum-Cortex
    49:  24,  # Right-Thalamus
    50:  25,  # Right-Caudate
    51:  26,  # Right-Putamen
    52:  27,  # Right-Pallidum
    53:  28,  # Right-Hippocampus
    54:  29,  # Right-Amygdala
    58:  30,  # Right-Accumbens-area
    60:  31,  # Right-VentralDC
    63:  32,  # Right-choroid-plexus
    77:  33,  # WM-hypointensities
    # --- Left cortical (aparc lh: 1001-1035) ---
    1002: 34,  # caudalanteriorcingulate-lh
    1003: 35,  # caudalmiddlefrontal-lh
    1005: 36,  # cuneus-lh
    1006: 37,  # entorhinal-lh
    1007: 38,  # fusiform-lh
    1008: 39,  # inferiorparietal-lh
    1009: 40,  # inferiortemporal-lh
    1010: 41,  # isthmuscingulate-lh
    1011: 42,  # lateraloccipital-lh
    1012: 43,  # lateralorbitofrontal-lh
    1013: 44,  # lingual-lh
    1014: 45,  # medialorbitofrontal-lh
    1015: 46,  # middletemporal-lh
    1016: 47,  # parahippocampal-lh
    1017: 48,  # paracentral-lh
    1018: 49,  # parsopercularis-lh
    1019: 50,  # parsorbitalis-lh
    1020: 51,  # parstriangularis-lh
    1021: 52,  # pericalcarine-lh
    1022: 53,  # postcentral-lh
    1023: 54,  # posteriorcingulate-lh
    1024: 55,  # precentral-lh
    1025: 56,  # precuneus-lh
    1026: 57,  # rostralanteriorcingulate-lh
    1027: 58,  # rostralmiddlefrontal-lh
    1028: 59,  # superiorfrontal-lh
    1029: 60,  # superiorparietal-lh
    1030: 61,  # superiortemporal-lh
    1031: 62,  # supramarginal-lh
    1034: 63,  # transversetemporal-lh
    1035: 64,  # insula-lh
    # --- Right cortical: rh-only labels → 65-78 ---
    2002: 65,  # caudalanteriorcingulate-rh
    2005: 66,  # cuneus-rh
    2010: 67,  # isthmuscingulate-rh
    2012: 68,  # lateralorbitofrontal-rh
    2013: 69,  # lingual-rh
    2014: 70,  # medialorbitofrontal-rh
    2016: 71,  # parahippocampal-rh
    2017: 72,  # paracentral-rh
    2021: 73,  # pericalcarine-rh
    2022: 74,  # postcentral-rh
    2023: 75,  # posteriorcingulate-rh
    2024: 76,  # precentral-rh
    2025: 77,  # precuneus-rh
    2028: 78,  # superiorfrontal-rh
    # --- Bilateral cortical: rh → same FastSurfer label as lh ---
    2003: 35,  # caudalmiddlefrontal-rh → 35 (bilateral)
    2006: 37,  # entorhinal-rh → 37
    2007: 38,  # fusiform-rh → 38
    2008: 39,  # inferiorparietal-rh → 39
    2009: 40,  # inferiortemporal-rh → 40
    2011: 42,  # lateraloccipital-rh → 42
    2015: 46,  # middletemporal-rh → 46
    2018: 49,  # parsopercularis-rh → 49
    2019: 50,  # parsorbitalis-rh → 50
    2020: 51,  # parstriangularis-rh → 51
    2026: 57,  # rostralanteriorcingulate-rh → 57
    2027: 58,  # rostralmiddlefrontal-rh → 58
    2029: 60,  # superiorparietal-rh → 60
    2030: 61,  # superiortemporal-rh → 61
    2031: 62,  # supramarginal-rh → 62
    2034: 63,  # transversetemporal-rh → 63
    2035: 64,  # insula-rh → 64

    # --- External / extra-cerebral CSF -> FastSurfer 79 ---
    # aparc+aseg.mgz에서 두개골 외부/경계의 non-brain CSF에 해당하는 레이블.
    # FreeSurfer 버전/subject마다 존재 여부가 다르므로 안전하게 모두 등록.
    # 258: 79,  # Extra-Cerebral  (aseg.auto_noCCseg 기반 두개골 외부 CSF; 가장 흔함)
    # 257: 79,  # Dura            (경막; 뇌 표면 바로 외부)
    # 165: 79,  # Skull           (일부 파이프라인에서 extra-cerebral로 활용)
    # 161: 79,  # unknown         (extra-cerebral non-brain tissue)
    # Note: label 24 (CSF)는 이미 15번(내부/sulcal CSF)으로 매핑됨.
}

FASTSURFER_LABELS = OrderedDict([
    (1,  "Cortical-WM-lh"),        (2,  "Lat-Ventricle-lh"),
    (3,  "Inf-Lat-Ventricle-lh"),  (4,  "Cerebellar-WM-lh"),
    (5,  "Cerebellar-Cortex-lh"),  (6,  "Thalamus-lh"),
    (7,  "Caudate-lh"),            (8,  "Putamen-lh"),
    (9,  "Pallidum-lh"),           (10, "3rd-Ventricle"),
    (11, "4th-Ventricle"),         (12, "Brain-Stem"),
    (13, "Hippocampus-lh"),        (14, "Amygdala-lh"),
    (15, "CSF"),                   (16, "Accumbens-lh"),
    (17, "Ventral-DC-lh"),         (18, "Choroid-Plexus-lh"),
    (19, "Cortical-WM-rh"),        (20, "Lat-Ventricle-rh"),
    (21, "Inf-Lat-Ventricle-rh"),  (22, "Cerebellar-WM-rh"),
    (23, "Cerebellar-Cortex-rh"),  (24, "Thalamus-rh"),
    (25, "Caudate-rh"),            (26, "Putamen-rh"),
    (27, "Pallidum-rh"),           (28, "Hippocampus-rh"),
    (29, "Amygdala-rh"),           (30, "Accumbens-rh"),
    (31, "Ventral-DC-rh"),         (32, "Choroid-Plexus-rh"),
    (33, "WM-hypointensities"),
    (34, "caudalanteriorcingulate-lh"), (35, "caudalmiddlefrontal(bi)"),
    (36, "cuneus-lh"),             (37, "entorhinal(bi)"),
    (38, "fusiform(bi)"),          (39, "inferiorparietal(bi)"),
    (40, "inferiortemporal(bi)"),  (41, "isthmuscingulate-lh"),
    (42, "lateraloccipital(bi)"),  (43, "lateralorbitofrontal-lh"),
    (44, "lingual-lh"),            (45, "medialorbitofrontal-lh"),
    (46, "middletemporal(bi)"),    (47, "parahippocampal-lh"),
    (48, "paracentral-lh"),        (49, "parsopercularis(bi)"),
    (50, "parsorbitalis(bi)"),     (51, "parstriangularis(bi)"),
    (52, "pericalcarine-lh"),      (53, "postcentral-lh"),
    (54, "posteriorcingulate-lh"), (55, "precentral-lh"),
    (56, "precuneus-lh"),          (57, "rostralanteriorcingulate(bi)"),
    (58, "rostralmiddlefrontal(bi)"), (59, "superiorfrontal-lh"),
    (60, "superiorparietal(bi)"),  (61, "superiortemporal(bi)"),
    (62, "supramarginal(bi)"),     (63, "transversetemporal(bi)"),
    (64, "insula(bi)"),
    (65, "caudalanteriorcingulate-rh"), (66, "cuneus-rh"),
    (67, "isthmuscingulate-rh"),   (68, "lateralorbitofrontal-rh"),
    (69, "lingual-rh"),            (70, "medialorbitofrontal-rh"),
    (71, "parahippocampal-rh"),    (72, "paracentral-rh"),
    (73, "pericalcarine-rh"),      (74, "postcentral-rh"),
    (75, "posteriorcingulate-rh"), (76, "precentral-rh"),
    (77, "precuneus-rh"),          (78, "superiorfrontal-rh"),
    # (79, "external-CSF"),
])


def classify_hemisphere(fs_label):
    left_only = {1,2,3,4,5,6,7,8,9,13,14,16,17,18,
                 34,36,41,43,44,45,47,48,52,53,54,55,56,59}
    right_only = {19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                  65,66,67,68,69,70,71,72,73,74,75,76,77,78}
    midline = {10, 11, 12, 15, 33}
    bilateral = {35,37,38,39,40,42,46,49,50,51,57,58,60,61,62,63,64}
    if fs_label in left_only: return "left"
    if fs_label in right_only: return "right"
    if fs_label in midline: return "midline"
    if fs_label in bilateral: return "bilateral"
    return "unknown"


def remap_volume(volume):
    max_fs = int(volume.max())
    lut = np.zeros(max_fs + 1, dtype=np.int32)
    for fs_label, fast_label in FREESURFER_TO_FASTSURFER.items():
        if fs_label <= max_fs:
            lut[fs_label] = fast_label
    remapped = lut[np.clip(volume, 0, max_fs)]
    n_total = (volume > 0).sum()
    n_mapped = (remapped > 0).sum()
    print(f"  Remap: {n_mapped}/{n_total} voxels ({n_mapped/n_total*100:.1f}%)")
    # External CSF 진단: 실제 atlas에서 몇 복셀이 79번으로 매핑됐는지 출력#######
    # _ext_csf_fs_labels = [258, 257, 165, 161]
    # _ext_csf_count = sum(int((volume == lb).sum()) for lb in _ext_csf_fs_labels)
    # if _ext_csf_count > 0:
    #    print(f"  External CSF (label 79): {_ext_csf_count} source voxels found "
    #          f"(FS labels {[lb for lb in _ext_csf_fs_labels if (volume==lb).any()]})")
    #else:
    #    print("  External CSF (label 79): NOT found in this atlas "
    #          "(labels 258/257/165/161 absent — label 79 will be empty)")
    ###########################
    present = set(np.unique(remapped)) - {0}
    missing = set(range(1, 79)) - present  # range(1, 79) → range(1, 80) to include label 79
    print(f"  Present: {len(present)}/78 labels")
    if missing:
        names = [FASTSURFER_LABELS.get(m, f"?{m}") for m in sorted(missing)]
        print(f"  Missing ({len(missing)}): {', '.join(names[:8])}{'...' if len(names)>8 else ''}")
    return remapped


def get_offsets(connectivity):
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0: continue
                m = abs(dx) + abs(dy) + abs(dz)
                if (connectivity == 6 and m == 1) or \
                   (connectivity == 18 and m <= 2) or \
                   connectivity == 26:
                    offsets.append((dx, dy, dz))
    return offsets


def compute_adjacency(volume, num_fg, connectivity=6):
    C = num_fg
    contact = np.zeros((C, C), dtype=np.int64)
    D, H, W = volume.shape
    print(f"  Computing {connectivity}-conn on {D}x{H}x{W}...")
    vol_idx = volume.astype(np.int32) - 1  # label k → index k-1, bg → -1
    for dx, dy, dz in get_offsets(connectivity):
        s_d = slice(max(0,-dx), D+min(0,-dx))
        s_h = slice(max(0,-dy), H+min(0,-dy))
        s_w = slice(max(0,-dz), W+min(0,-dz))
        t_d = slice(max(0,dx), D+min(0,dx))
        t_h = slice(max(0,dy), H+min(0,dy))
        t_w = slice(max(0,dz), W+min(0,dz))
        src = vol_idx[s_d,s_h,s_w]; tgt = vol_idx[t_d,t_h,t_w]
        bnd = (src >= 0) & (tgt >= 0) & (src != tgt)
        if bnd.any():
            np.add.at(contact, (src[bnd], tgt[bnd]), 1)
    contact = (contact + contact.T) // 2
    np.fill_diagonal(contact, 0)
    adj_bin = (contact > 0)
    mx = contact.max()
    adj_str = contact.astype(np.float64) / mx if mx > 0 else contact.astype(np.float64)
    print(f"  ✓ {adj_bin.sum()//2} edges, density={adj_bin.sum()//2/(C*(C-1)/2):.3f}")
    return adj_bin, adj_str, contact


def build_atlas_prior(atlas_path, connectivity=6, min_contact=5):
    NUM_FG = 78 # 79
    print("=" * 60)
    print("Building Atlas Graph Prior (FastSurfer 78-class)")
    print("=" * 60)
    img = nib.load(atlas_path)
    volume = np.asarray(img.dataobj, dtype=np.int32)
    print(f"✓ Atlas: shape={volume.shape}, voxel={img.header.get_zooms()[:3]}")
    remapped = remap_volume(volume)
    adj_bin, adj_str, raw = compute_adjacency(remapped, NUM_FG, connectivity)
    # Threshold
    adj_bin_t = (raw >= min_contact)
    adj_str_t = adj_str.copy()
    adj_str_t[raw < min_contact] = 0.0
    mx = adj_str_t.max()
    if mx > 0: adj_str_t /= mx
    n0 = adj_bin.sum()//2; n1 = adj_bin_t.sum()//2
    print(f"  Threshold(>={min_contact}): {n0} → {n1} edges")
    hemi = {i: classify_hemisphere(i) for i in range(1,79)}
    result = {
        "adjacency_binary": adj_bin_t.astype(np.float32),
        "adjacency_strength": adj_str_t.astype(np.float32),
        "possible_mask": adj_bin_t.astype(np.float32),
        "raw_contact_counts": raw.astype(np.float32),
        "num_classes": 79, "num_foreground": NUM_FG,
        "label_names": dict(FASTSURFER_LABELS),
        "hemisphere_map": hemi,
        "config": {"atlas_path": str(atlas_path), "connectivity": connectivity,
                   "min_contact_voxels": min_contact},
        "freesurfer_to_fastsurfer_map": FREESURFER_TO_FASTSURFER,
    }
    print(f"\nSummary: 79 ROIs, {n1} edges, density={n1/(NUM_FG*(NUM_FG-1)/2):.3f}")
    return result


def to_torch(prior):
    import torch
    out = dict(prior)
    for k in ["adjacency_binary","adjacency_strength","possible_mask","raw_contact_counts"]:
        if isinstance(out[k], np.ndarray):
            out[k] = torch.from_numpy(out[k])
    return out


def get_prior_for_training(prior_path):
    """Drop-in replacement for get_priors_for_num_classes(80)."""
    import torch
    p = torch.load(prior_path, weights_only=False)
    pm = p["possible_mask"]; es = p["adjacency_strength"]
    if isinstance(pm, np.ndarray): pm = torch.from_numpy(pm)
    if isinstance(es, np.ndarray): es = torch.from_numpy(es)
    return pm.float(), es.float()


def visualize(prior, output_path):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음"); return
    adj = prior["adjacency_binary"]; strength = prior["adjacency_strength"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].imshow(adj, cmap='Blues', aspect='equal')
    axes[0].set_title(f"Binary Adjacency ({int(adj.sum())//2} edges)")
    im = axes[1].imshow(strength, cmap='YlOrRd', aspect='equal')
    axes[1].set_title("Contact Strength (normalized)")
    plt.colorbar(im, ax=axes[1], shrink=0.6)
    for ax in axes:
        for b in [18, 33, 64]:
            ax.axhline(b-0.5, color='red', lw=0.8, alpha=0.6)
            ax.axvline(b-0.5, color='red', lw=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}"); plt.close()
    # Top connections
    C = adj.shape[0]; names = prior["label_names"]
    pairs = [(strength[i,j],i,j) for i in range(C) for j in range(i+1,C) if strength[i,j]>0]
    pairs.sort(reverse=True)
    print(f"\n  Top 20 strongest:")
    for r,(s,i,j) in enumerate(pairs[:20]):
        print(f"    {r+1:2d}. {names.get(i+1,'?'):32s} ↔ {names.get(j+1,'?'):32s}  s={s:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output", default="atlas_prior_78class.pt")
    parser.add_argument("--connectivity", type=int, default=6, choices=[6,18,26])
    parser.add_argument("--min-contact", type=int, default=5)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    prior = build_atlas_prior(args.atlas, args.connectivity, args.min_contact)
    try:
        import torch; torch.save(to_torch(prior), args.output)
    except ImportError:
        import pickle
        with open(args.output,'wb') as f: pickle.dump(prior, f)
    print(f"\n✓ Saved: {args.output}")
    if args.visualize:
        visualize(prior, str(Path(args.output).with_suffix('.png')))
    print(f"\n--- Usage ---")
    print(f"from build_atlas_prior import get_prior_for_training")
    print(f"possible_mask, expected_strength = get_prior_for_training('{args.output}')")
    print(f"# Both [78, 78] — drop-in for get_78class_hard_prior()")