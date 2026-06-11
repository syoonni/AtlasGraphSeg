"""
nnU-Net 모델 래퍼 - pretrained weights 로드 및 fine-tuning 지원
"""
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from batchgenerators.utilities.file_and_folder_operations import load_json
# nnU-Net 경로 추가
#NNUNET_PATH = Path("/home/hwlim/syoon/nnUNet")
#sys.path.insert(0, str(NNUNET_PATH))

# nnU-Net v2 imports
try:
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.paths import nnUNet_results
    NNUNET_V2_AVAILABLE = True
except ImportError as e:
    NNUNET_V2_AVAILABLE = False
    print(f"Warning: nnUNet v2 import failed: {e}")


class nnUNetWrapper(nn.Module):
    """
    nnU-Net 모델을 래핑하여 fine-tuning 가능하게 만듦.
    
    두 가지 모드:
    1. From scratch: nnU-Net 아키텍처만 사용
    2. Pretrained: 학습된 weights 로드 후 fine-tune
    """
    
    def __init__(
        self,
        pretrained_path: str,
        device: str = 'cuda',
    ):
        super().__init__()

        if not pretrained_path:
            raise ValueError(
                "pretrained_path must be provided. "
                "nnUNetWrapper requires a trained nnU-Net checkpoint."
            )
        if not Path(pretrained_path).exists():
            raise FileNotFoundError(f"pretrained_path does not exist: {pretrained_path}")

        self.device = device

        self.model = self._load_pretrained(pretrained_path)
        self.model = self.model.to(device)

    # ------------------------------------------------------------------
    # 로드
    # ------------------------------------------------------------------            
    def _load_pretrained(self, checkpoint_path: str) -> nn.Module:
        """
        Pretrained nnU-Net checkpoint 로드.
        plans.json / dataset.json이 없으면 바로 error.
        """
        checkpoint_path = Path(checkpoint_path)

        # ckpt 파일 경로 결정
        if checkpoint_path.is_file():
            ckpt_file = checkpoint_path
            model_folder = checkpoint_path.parent.parent   # fold_X/ 위로 두 단계
        else:
            model_folder = checkpoint_path
            ckpt_file = model_folder / "fold_0" / "checkpoint_best.pth"
            if not ckpt_file.exists():
                ckpt_file = model_folder / "fold_0" / "checkpoint_final.pth"

        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        # plans.json / dataset.json 존재 확인
        plans_file       = model_folder / "plans.json"
        dataset_json_file = model_folder / "dataset.json"

        missing = []
        if not plans_file.exists():
            missing.append(str(plans_file))
        if not dataset_json_file.exists():
            missing.append(str(dataset_json_file))
        if missing:
            raise FileNotFoundError(
                f"nnU-Net plans/dataset.json not found in {model_folder}. "
                f"Missing: {missing}"
            )

        # checkpoint → state_dict
        print(f"Loading checkpoint: {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location='cpu')

        if 'network_weights' in checkpoint:
            state_dict = checkpoint['network_weights']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 아키텍처 빌드 → weights 로드
        plans        = load_json(str(plans_file))
        dataset_json = load_json(str(dataset_json_file))

        model = self._build_network_from_plans(plans, dataset_json)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded pretrained weights successfully")

        return model
    
    # ------------------------------------------------------------------
    # nnU-Net 아키텍처 빌드 (plans 기반, fallback 없음)
    # ------------------------------------------------------------------
    def _build_network_from_plans(self, plans: dict, dataset_json: dict) -> nn.Module:
        """
        nnU-Net v2.5+ API를 사용하여 네트워크 구축.
        
        New API signature:
        get_network_from_plans(
            arch_class_name, arch_kwargs, arch_kwargs_req_import,
            input_channels, output_channels, 
            allow_init=True, deep_supervision=None
        )
        """
        if not NNUNET_V2_AVAILABLE:
            raise ImportError(
                "nnUNet v2 is not installed or not importable. "
                "Cannot reconstruct the network architecture from plans. "
                f"Checked path: {NNUNET_PATH}"
            )

        plans_manager = PlansManager(plans)
        configuration = "3d_fullres"
        
        # Get configuration details
        config_manager = plans_manager.get_configuration(configuration)
        
        # Extract architecture information from plans
        # New API requires these specific parameters
        arch_class_name = config_manager.network_arch_class_name
        arch_kwargs = config_manager.network_arch_init_kwargs
        arch_kwargs_req_import = config_manager.network_arch_init_kwargs_req_import
        
        # Get input/output channels
        input_channels = len(dataset_json['channel_names'])
        
        # Get number of output channels (segmentation classes)
        # This is stored in plans under label manager info
        label_manager = plans_manager.get_label_manager(dataset_json)
        output_channels = label_manager.num_segmentation_heads
        self.num_classes = output_channels
        
        print(f"Building network: {arch_class_name}")
        print(f"  Input channels: {input_channels}")
        print(f"  Output channels: {output_channels}")

        model = get_network_from_plans(
            arch_class_name=arch_class_name,
            arch_kwargs=arch_kwargs,
            arch_kwargs_req_import=arch_kwargs_req_import,
            input_channels=input_channels,
            output_channels=output_channels,
            allow_init=True,
            deep_supervision=True,
        )
        
        return model

    # ------------------------------------------------------------------
    # forward / utility
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        deep supervision 출력이면 가장 높은 해상도만 반환.

        Args:
            x: [B, C, D, H, W]
        Returns:
            logits: [B, num_classes, D, H, W]
        """
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output

    def freeze_encoder(self, freeze: bool = True):
        """Encoder 부분만 freeze (fine-tuning용)"""
        for name, param in self.model.named_parameters():
            if any(x in name for x in ['encoder', 'conv_blocks_context', 'td']):
                param.requires_grad = not freeze

        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.parameters())
        print(f"Frozen {frozen}/{total} parameters")