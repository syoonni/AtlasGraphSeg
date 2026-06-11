from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import warnings
import re

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    nib = None

class FastSurferLabelMap:
    """FastSurfer atlas 정보 - 제공된 테이블 기반의 정확한 매핑"""

    FASTSURFER_LABELS = {
        0: "Background",
        
        # Subcortical Structures (1-33)
        1: "Cortical-white-matter-lh",
        2: "Lateral-Ventricle-lh",
        3: "Inferior-Lateral-Ventricle-lh", 
        4: "Cerebellar-White-Matter-lh",
        5: "Cerebellar-Cortex-lh",
        6: "Thalamus-lh",
        7: "Caudate-lh",
        8: "Putamen-lh",
        9: "Pallidum-lh",
        10: "3rd-Ventricle",
        11: "4th-Ventricle",
        12: "Brain-Stem",
        13: "Hippocampus-lh",
        14: "Amygdala-lh",
        15: "CSF",
        16: "Accumbens-lh",
        17: "Ventral-DC-lh",
        18: "Choroid-Plexus-lh",
        19: "Cortical-white-matter-rh",
        20: "Lateral-Ventricle-rh", 
        21: "Inferior-Lateral-Ventricle-rh",
        22: "Cerebellar-White-Matter-rh",
        23: "Cerebellar-Cortex-rh",
        24: "Thalamus-rh",
        25: "Caudate-rh",
        26: "Putamen-rh",
        27: "Pallidum-rh",
        28: "Hippocampus-rh",
        29: "Amygdala-rh",
        30: "Accumbens-rh",
        31: "Ventral-DC-rh",
        32: "Choroid-Plexus-rh",
        33: "WM-hypointensities",
        
        # Cortical Structures (34-78)
        34: "caudalanteriorcingulate-lh",
        35: "caudalmiddlefrontal-lh",  # (lh, rh) - bilateral
        36: "cuneus-lh",
        37: "entorhinal-lh",  # (lh, rh) - bilateral
        38: "fusiform-lh",  # (lh, rh) - bilateral
        39: "inferiorparietal-lh",  # (lh, rh) - bilateral
        40: "inferiortemporal-lh",  # (lh, rh) - bilateral
        41: "isthmuscingulate-lh",
        42: "lateraloccipital-lh",  # (lh, rh) - bilateral
        43: "lateralorbitofrontal-lh",
        44: "lingual-lh",
        45: "medialorbitofrontal-lh",
        46: "middletemporal-lh",  # (lh, rh) - bilateral
        47: "parahippocampal-lh",
        48: "paracentral-lh",
        49: "parsopercularis-lh",  # (lh, rh) - bilateral
        50: "parsorbitalis-lh",  # (lh, rh) - bilateral
        51: "parstriangularis-lh",  # (lh, rh) - bilateral
        52: "pericalcarine-lh",
        53: "postcentral-lh",
        54: "posteriorcingulate-lh",
        55: "precentral-lh",
        56: "precuneus-lh",
        57: "rostralanteriorcingulate-lh",  # (lh, rh) - bilateral
        58: "rostralmiddlefrontal-lh",  # (lh, rh) - bilateral
        59: "superiorfrontal-lh",
        60: "superiorparietal-lh",  # (lh, rh) - bilateral
        61: "superiortemporal-lh",  # (lh, rh) - bilateral
        62: "supramarginal-lh",  # (lh, rh) - bilateral
        63: "transversetemporal-lh",  # (lh, rh) - bilateral
        64: "insula-lh",  # (lh, rh) - bilateral
        
        # Right Hemisphere Cortical (65-78)
        65: "caudalanteriorcingulate-rh",
        66: "cuneus-rh",
        67: "isthmuscingulate-rh",
        68: "lateralorbitofrontal-rh",
        69: "lingual-rh",
        70: "medialorbitofrontal-rh",
        71: "parahippocampal-rh",
        72: "paracentral-rh",
        73: "pericalcarine-rh",
        74: "postcentral-rh",
        75: "posteriorcingulate-rh",
        76: "precentral-rh",
        77: "precuneus-rh",
        78: "superiorfrontal-rh"
        #79: "external-CSF"
    }

    # Simplified label mapping schemes for FastSurfer
    SIMPLIFIED_5_CLASS = {
        'background': 0,
        'cortical_gray': 1,
        'subcortical_gray': 2,
        'white_matter': 3,
        'csf': 4,
    }

    # FastSurfer to simplified mapping (updated for correct structure)
    FASTSURFER_TO_SIMPLIFIED = {
        0: 0,  # Background
        
        # White matter structures -> white_matter (3)
        1: 3,   # Cortical-white-matter-lh
        19: 3,  # Cortical-white-matter-rh
        4: 3,   # Cerebellar-White-Matter-lh
        22: 3,  # Cerebellar-White-Matter-rh
        33: 3,  # WM-hypointensities
        
        # CSF and ventricles -> csf (4)
        2: 4,   # Lateral-Ventricle-lh
        3: 4,   # Inferior-Lateral-Ventricle-lh
        20: 4,  # Lateral-Ventricle-rh
        21: 4,  # Inferior-Lateral-Ventricle-rh
        10: 4,  # 3rd-Ventricle
        11: 4,  # 4th-Ventricle
        15: 4,  # CSF
        18: 4,  # Choroid-Plexus-lh
        32: 4,  # Choroid-Plexus-rh
        
        # Subcortical gray matter -> subcortical_gray (2)
        6: 2,   # Thalamus-lh
        7: 2,   # Caudate-lh
        8: 2,   # Putamen-lh
        9: 2,   # Pallidum-lh
        13: 2,  # Hippocampus-lh
        14: 2,  # Amygdala-lh
        16: 2,  # Accumbens-lh
        17: 2,  # Ventral-DC-lh
        24: 2,  # Thalamus-rh
        25: 2,  # Caudate-rh
        26: 2,  # Putamen-rh
        27: 2,  # Pallidum-rh
        28: 2,  # Hippocampus-rh
        29: 2,  # Amygdala-rh
        30: 2,  # Accumbens-rh
        31: 2,  # Ventral-DC-rh
        12: 2,  # Brain-Stem
        
        # Cerebellar cortex -> cortical_gray (1)
        5: 1,   # Cerebellar-Cortex-lh
        23: 1,  # Cerebellar-Cortex-rh
    }
    
    # Add all cortical regions to cortical_gray (1)
    for i in range(34, 65):  # Left cortical regions (34-64)
        FASTSURFER_TO_SIMPLIFIED[i] = 1
    for i in range(65, 79):  # Right cortical regions (65-78)
        FASTSURFER_TO_SIMPLIFIED[i] = 1

    @classmethod
    def simplify_labels(cls, fastsurfer_labels: torch.Tensor) -> torch.Tensor:
        """
        Convert FastSurfer labels to simplified label set.
        """
        squeeze_batch = False
        if fastsurfer_labels.ndim == 3:
            fastsurfer_labels = fastsurfer_labels.unsqueeze(0)
            squeeze_batch = True
            
        simplified = torch.zeros_like(fastsurfer_labels)
        
        for fs_label, simple_label in cls.FASTSURFER_TO_SIMPLIFIED.items():
            simplified[fastsurfer_labels == fs_label] = simple_label
            
        if squeeze_batch:
            simplified = simplified.squeeze(0)
            
        return simplified
    
    @classmethod
    def simplify_labels_numpy(cls, fastsurfer_labels: np.ndarray) -> np.ndarray:
        """
        Convert FastSurfer labels to simplified label set (numpy version).
        """
        simplified = np.zeros_like(fastsurfer_labels)
        
        for fs_label, simple_label in cls.FASTSURFER_TO_SIMPLIFIED.items():
            simplified[fastsurfer_labels == fs_label] = simple_label
            
        return simplified
    
    @classmethod
    def get_structural_edges(cls) -> List[Tuple[int, int]]:
        """Get anatomically motivated structural edges for simplified labels."""
        edges = [
            (1, 2),  # cortical to subcortical
            (1, 3),  # cortical to white matter
            (2, 3),  # subcortical to white matter
            (3, 4),  # white matter to CSF (ventricular boundaries)
        ]
        return edges

    @classmethod
    def get_bilateral_symmetry_edges(cls) -> List[Tuple[int, int]]:
        """
        Get bilateral symmetry constraints for FastSurfer labels.
        Based on the correct anatomical correspondence.
        """
        symmetry_pairs = []
        
        # Subcortical symmetry (확실한 대칭 구조)
        subcortical_pairs = [
            (1, 19),   # Cortical-white-matter
            (2, 20),   # Lateral-Ventricle
            (3, 21),   # Inferior-Lateral-Ventricle
            (4, 22),   # Cerebellar-White-Matter
            (5, 23),   # Cerebellar-Cortex
            (6, 24),   # Thalamus
            (7, 25),   # Caudate
            (8, 26),   # Putamen
            (9, 27),   # Pallidum
            (13, 28),  # Hippocampus
            (14, 29),  # Amygdala
            (16, 30),  # Accumbens
            (17, 31),  # Ventral-DC
            (18, 32),  # Choroid-Plexus
        ]
        
        # Cortical symmetry (확실한 대응 관계만)
        cortical_pairs = [
            (34, 65),  # caudalanteriorcingulate
            (36, 66),  # cuneus
            (41, 67),  # isthmuscingulate
            (43, 68),  # lateralorbitofrontal
            (44, 69),  # lingual
            (45, 70),  # medialorbitofrontal
            (47, 71),  # parahippocampal
            (48, 72),  # paracentral
            (52, 73),  # pericalcarine
            (53, 74),  # postcentral
            (54, 75),  # posteriorcingulate
            (55, 76),  # precentral
            (56, 77),  # precuneus
            (59, 78),  # superiorfrontal
        ]
        
        return subcortical_pairs + cortical_pairs


# Dataset classes remain the same as they are compatible
class BrainSegmentationDataset(Dataset):
    """Dataset class for brain segmentation with FastSurfer integration."""
    
    def __init__(
        self,
        image_paths: List[Path],
        label_paths: Optional[List[Path]] = None,
        atlas_paths: Optional[List[Path]] = None,
        transform: Optional[callable] = None,
        use_simplified_labels: bool = False,
        patch_size: Optional[Tuple[int, int, int]] = None,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths or [None] * len(image_paths)
        self.atlas_paths = atlas_paths or [None] * len(image_paths)
        self.transform = transform
        self.use_simplified_labels = use_simplified_labels
        self.patch_size = patch_size
        
        if len(self.label_paths) != len(self.image_paths):
            raise ValueError("Number of labels must match number of images")
        if len(self.atlas_paths) != len(self.image_paths):
            raise ValueError("Number of atlas files must match number of images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image = self._load_volume(self.image_paths[idx])
        sample = {'image': image}
        
        # Load label if available
        if self.label_paths[idx] is not None:
            label = self._load_volume(self.label_paths[idx])
            sample['label'] = label

            if self.atlas_paths[idx] is None:
                sample['atlas'] = label.clone()
            else:
                atlas = self._load_volume(self.atlas_paths[idx])
                sample['atlas'] = atlas
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        # Extract patches if specified
        if self.patch_size is not None:
            sample = self._extract_patch(sample)
            
        return sample
    
    def _load_volume(self, path: Path) -> torch.Tensor:
        img = nib.load(path)
        data = img.get_fdata()
        return torch.from_numpy(data).float()
    
    def _extract_patch(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract random patch from full volume."""
        if self.patch_size is None:
            return sample
            
        # Implementation same as before
        image = sample['image']
        if image.ndim == 4:
            _, D, H, W = image.shape
        else:
            D, H, W = image.shape
            
        pd, ph, pw = self.patch_size
        start_d = torch.randint(0, max(1, D - pd), (1,)).item()
        start_h = torch.randint(0, max(1, H - ph), (1,)).item()  
        start_w = torch.randint(0, max(1, W - pw), (1,)).item()
        
        end_d = min(start_d + pd, D)
        end_h = min(start_h + ph, H)
        end_w = min(start_w + pw, W)
        
        patched_sample = {}
        for key, volume in sample.items():
            if volume.ndim == 4:
                patched_sample[key] = volume[:, start_d:end_d, start_h:end_h, start_w:end_w]
            else:
                patched_sample[key] = volume[start_d:end_d, start_h:end_h, start_w:end_w]
                
        return patched_sample


class BrainSegmentationTransforms:
    """Common transforms for brain segmentation data."""
    
    @staticmethod
    def normalize_intensity(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Z-score normalization of image intensity."""
        image = sample['image']
        mask = image > 0
        if mask.sum() > 0:
            mean_val = image[mask].mean()
            std_val = image[mask].std()
            image = (image - mean_val) / (std_val + 1e-8)
        sample['image'] = image
        return sample
    
    @staticmethod
    def add_channel_dim(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add channel dimension to image."""
        if sample['image'].ndim == 3:
            sample['image'] = sample['image'].unsqueeze(0)
        return sample
    
    @staticmethod
    def random_flip(sample: Dict[str, torch.Tensor], p: float = 0.5) -> Dict[str, torch.Tensor]:
        """Random left-right flip augmentation."""
        if torch.rand(1).item() < p:
            for key, volume in sample.items():
                if volume.ndim == 4:
                    sample[key] = torch.flip(volume, [-1])
                elif volume.ndim == 3:
                    sample[key] = torch.flip(volume, [-1])
        return sample
    
    @staticmethod
    def compose(*transforms) -> callable:
        """Compose multiple transforms."""
        def composed_transform(sample):
            for transform in transforms:
                sample = transform(sample)
            return sample
        return composed_transform


def create_synthetic_data(
    num_samples: int = 10,
    volume_size: Tuple[int, int, int] = (64, 64, 64),
    num_classes: int = 5,
    save_dir: Optional[Path] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Create synthetic brain data for testing and development."""
    samples = []
    D, H, W = volume_size
    
    for i in range(num_samples):
        # Create synthetic brain data
        image = torch.randn(1, D, H, W) * 0.2 + 0.5
        center = (D//2, H//2, W//2)
        Y, X, Z = torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'
        )
        dist = ((Y - center[0])**2 + (X - center[1])**2 + (Z - center[2])**2).sqrt()
        brain_mask = dist < min(D, H, W) // 3
        image = image * brain_mask.float().unsqueeze(0)
        
        label = torch.zeros(D, H, W, dtype=torch.long)
        if num_classes > 1:
            central_mask = dist < min(D, H, W) // 6
            label[central_mask] = 1
        if num_classes > 2:
            ring_mask = (dist >= min(D, H, W) // 6) & (dist < min(D, H, W) // 4)
            label[ring_mask] = 2
        if num_classes > 3:
            outer_mask = (dist >= min(D, H, W) // 4) & brain_mask
            label[outer_mask] = 3
        
        sample = {
            'image': image,
            'label': label,
            'atlas': label.clone(),
        }
        
        samples.append(sample)
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(sample, save_dir / f"sample_{i:03d}.pt")
    
    return samples

class FreeSurferDataset(Dataset):
    """
    ADNI MRI + FastSurfer Segmentation Dataset.
    
    Expects:
        - MRI files: {mri_dir}/MRimages_XXXX.nii.gz
        - Seg files: {seg_dir}/Segment_XXXX.nii.gz
    """
    
    def __init__(
        self,
        mri_dir: str,
        seg_dir: str,
        patch_size: Optional[Tuple[int, int, int]] = (112, 128, 128),
        use_simplified_labels: bool = False,
        augment: bool = False,
        normalize: bool = True,
        center_crop: bool = False,
    ):
        
        self.mri_dir = Path(mri_dir)
        self.seg_dir = Path(seg_dir)
        self.patch_size = patch_size
        self.use_simplified_labels = use_simplified_labels
        self.augment = augment
        self.normalize = normalize
        self.center_crop = center_crop
        
        self.pairs = self._find_pairs()
        print(f"Found {len(self.pairs)} MRI-Segmentation pairs")
        
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Match FreeSurfer MRI files with segmentation files."""
        pairs = []
        # 모든 nii.gz 파일을 찾습니다.
        mri_files = sorted(self.mri_dir.glob("oasis_*.nii.gz"))
        
        for mri_path in mri_files:
            match = re.search(r'(oasis_\d+)_\d+\.nii\.gz', mri_path.name)
            
            if match:
                subject_id = match.group(1)
                
                seg_path = self.seg_dir / f"{subject_id}.nii.gz"
                
                if seg_path.exists():
                    pairs.append((mri_path, seg_path))
                else:
                    print(f"Warning: No segmentation for {mri_path.name} (Expected: {seg_path.name})")
                    
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mri_path, seg_path = self.pairs[idx]
        
        # Load NIfTI files
        mri_data = nib.load(str(mri_path)).get_fdata().astype(np.float32)
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.int64)

        mri_data = mri_data.transpose(2, 1, 0) # 또는 (2, 1, 0)
        seg_data = seg_data.transpose(2, 1, 0) # MRI와 똑같이 적용
        
        # Simplify labels
        if self.use_simplified_labels:
            seg_data = FastSurferLabelMap.simplify_labels_numpy(seg_data)
        
        # Normalize MRI intensity
        if self.normalize:
            brain_mask = seg_data > 0
            if brain_mask.sum() > 0:
                mean_val = mri_data[brain_mask].mean()
                std_val = mri_data[brain_mask].std() + 1e-8
                mri_data = (mri_data - mean_val) / std_val
        
        # Extract patch if specified
        if self.patch_size is not None:
            mri_data, seg_data = self._extract_patch(mri_data, seg_data)
        
        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            mri_data = np.flip(mri_data, axis=2).copy()
            seg_data = np.flip(seg_data, axis=2).copy()
        
        # Convert to tensors
        image = torch.from_numpy(mri_data).float().unsqueeze(0)  # [1, D, H, W]
        label = torch.from_numpy(seg_data).long()  # [D, H, W]
        
        return {
            'image': image,
            'label': label,
            'atlas': label.clone(),  # Use GT as atlas
        }
    
    def _extract_patch(
        self, 
        mri: np.ndarray, 
        seg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random patch from volume."""
        D, H, W = mri.shape
        pd, ph, pw = self.patch_size
        
        if self.center_crop:
            d_start = max(0, (D - pd) // 2)
            h_start = max(0, (H - ph) // 2)
            w_start = max(0, (W - pw) // 2)
        # Foreground oversampling (nnU-Net 방식)
        # if np.random.rand() < 0.5 and (seg > 0).any():
        #    fg_indices = np.argwhere(seg > 0)
        #    center = fg_indices[np.random.randint(len(fg_indices))]
        #    d_start = int(np.clip(center[0] - pd//2, 0, max(0, D - pd)))
        #    h_start = int(np.clip(center[1] - ph//2, 0, max(0, H - ph)))
        #    w_start = int(np.clip(center[2] - pw//2, 0, max(0, W - pw)))
        else:
            d_start = np.random.randint(0, max(1, D - pd + 1))
            h_start = np.random.randint(0, max(1, H - ph + 1))
            w_start = np.random.randint(0, max(1, W - pw + 1))
        
        mri_patch = mri[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        seg_patch = seg[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        # Pad if necessary
        if mri_patch.shape != self.patch_size:
            pad_d = self.patch_size[0] - mri_patch.shape[0]
            pad_h = self.patch_size[1] - mri_patch.shape[1]
            pad_w = self.patch_size[2] - mri_patch.shape[2]
            mri_patch = np.pad(mri_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            seg_patch = np.pad(seg_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            
        return mri_patch, seg_patch