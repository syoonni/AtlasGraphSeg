"""
nnU-Net + Graph Prior 통합 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from torch.cuda.amp import autocast, GradScaler


from .nnunet_wrapper import nnUNetWrapper
from loss.hard_loss import HardPriorLoss
from loss.island_penalty import IslandPenaltyLoss
from loss.total_loss import compute_all_losses
from graph import soft_adjacency_from_probs, hard_adjacency_from_mask
from metrics import compute_structural_score, compute_per_class_dice


class GraphEnhancednnUNet(nn.Module):
    """
    nnU-Net backbone + Graph Prior Loss를 함께 사용하는 모델.
    
    학습 시: segmentation loss + graph prior loss
    추론 시: 일반 segmentation과 동일
    """
    
    def __init__(
        self,
        nnunet_model: nnUNetWrapper,
        num_classes: int = 5,
        connectivity: str = '6-connectivity',
        lambda_graph: float = 0.0,
        lambda_island: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        
        self.backbone = nnunet_model
        self.num_classes = num_classes
        self.connectivity = connectivity
        
        # NEW: Store lambdas
        self.lambda_hard = lambda_graph
        self.lambda_island = lambda_island
        
        # Initialize loss modules
        self.hard_loss_fn = HardPriorLoss(num_classes)
        self.island_loss_fn = IslandPenaltyLoss()
        
        print(f"GraphEnhancednnUNet initialized:")
        print(f"  num_classes: {num_classes}")
        print(f"  connectivity: {connectivity}")
        print(f"  lambda_graph: {lambda_graph} → lambda_hard={self.lambda_hard}")
        print(f"  lambda_island: {lambda_island}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - logits만 반환 (추론용)"""
        return self.backbone(x)
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        is_train: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        전체 loss 계산.
        """
        return compute_all_losses(
            logits=logits,
            target=target,
            num_classes=self.num_classes,
            connectivity=self.connectivity,
            hard_loss_fn=self.hard_loss_fn,
            island_loss_fn=self.island_loss_fn,
            is_train=is_train,
        )

class GraphPriorTrainer:
    """
    Graph Prior가 추가된 nnU-Net 학습을 관리하는 Trainer.
    """

    # Key structures to evaluate topology/surface on (subset for speed)
    # These are the most clinically important structures
    KEY_STRUCTURES = [
        5, 6, 7, 8, 12, 13, 14,          # LH: cerebellum-ctx, thalamus, caudate, putamen, brainstem, hippo, amygdala
        23, 24, 25, 26, 28, 29,           # RH: cerebellum-ctx, thalamus, caudate, putamen, hippo, amygdala
    ]
    
    def __init__(
        self,
        model: GraphEnhancednnUNet,
        train_loader,
        val_loader,
        lr: float = 1e-4,
        device: str = 'cuda',
        num_classes: int = 79,
        voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        eval_topology: bool = True,
        eval_surfaces: bool = True,
        eval_all_classes_surface: bool = False,
    ):
        self.model = model.to(device)
        if torch.cuda.device_count() > 1:
            self.model_parallel = torch.nn.DataParallel(model)
        else:
            self.model_parallel = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.voxel_spacing = voxel_spacing

        # Evaluation settings
        self.eval_topology = eval_topology
        self.eval_surfaces = eval_surfaces
        self.eval_all_classes_surface = eval_all_classes_surface
        
        self.lambda_hard = model.lambda_hard
        self.lambda_island = model.lambda_island
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )

        self.scaler = GradScaler()

        # Training state
        self.current_epoch = 0
        self.best_dice = 0.0
        self.best_loss = float('inf')
        self.is_full_eval = False  # Whether to compute expensive metrics this epoch
        
        print(f"Trainer initialized:")
        print(f"  lr: {lr}, device: {device}")
        print(f"  lambda_hard: {self.lambda_hard}")
        print(f"  lambda_island: {self.lambda_island}")
        print(f"  voxel_spacing: {self.voxel_spacing}")
        print(f"  eval_topology: {self.eval_topology}")
        print(f"  eval_surfaces: {self.eval_surfaces}")
        
    def train_epoch(self) -> Dict[str, float]:
        """한 epoch 학습"""
        self.model.train()
        accum = {'seg': 0.0, 'hard': 0.0, 'island': 0.0, 'total': 0.0}
        n_batches = 0
        
        for batch in self.train_loader:
            image = batch['image'].to(self.device)
            label = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()

            with autocast():
                # logits = self.model(image)
                logits = self.model_parallel(image)
                losses = self.model.compute_loss(logits.float(), label, is_train=True)
                train_loss = (losses['seg']
                              + self.lambda_hard * losses['hard']
                              + self.lambda_island * losses['island']) # train일때는 island 포함 x -> lambda_island * 0
                
            if torch.isnan(train_loss) or torch.isinf(train_loss):
                self.optimizer.zero_grad()
                continue
            
            self.scaler.scale(train_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 첫 배치만 디버그 출력 (backward 후라 grad_norm 유효)
            if n_batches == 0:
                print(f"  [Batch 0] seg={losses['seg'].item():.4f} | "
                      f"hard={losses['hard'].item():.4f} | "
                      f"total={train_loss.item():.4f} | "
                      f"grad_norm={grad_norm:.4f} | "
                      f"scaler={self.scaler.get_scale():.0f}")
            
            accum['seg'] += losses['seg'].item()
            accum['hard'] += losses['hard'].item()
            accum['island'] += losses['island'].item()    # 로깅만
            accum['total'] += train_loss.item()
            n_batches += 1           
            
        self.current_epoch += 1
        return {k: v / max(n_batches, 1) for k, v in accum.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        np.random.seed(42)   
        torch.manual_seed(42) 
        
        accum = {'seg': 0.0, 'hard': 0.0, 'island': 0.0, 'total': 0.0}
        dice_scores = []
        struct_accum = {}

        # Determine which classes to evaluate for expensive metrics
        surface_classes = None if self.eval_all_classes_surface else self.KEY_STRUCTURES
        topology_classes = self.KEY_STRUCTURES  # Always use subset for topology (very slow otherwise)
        
        n_batches = 0
        
        for batch in self.val_loader:
            image = batch['image'].to(self.device)
            label = batch['label'].to(self.device)
            
            logits = self.model(image).float()
            
            # ── Loss 계산 — compute_all_losses 사용 ──
            losses = self.model.compute_loss(logits, label, is_train=False)
            val_loss = (losses['seg']
                        + self.lambda_hard * losses['hard']
                        + self.lambda_island * losses['island'])
            
            accum['seg'] += losses['seg'].item()
            accum['hard'] += losses['hard'].item()
            accum['island'] += losses['island'].item()
            accum['total'] += val_loss.item()
            
            pred = logits.argmax(dim=1)
            # ── Dice ──
            dice_result = compute_per_class_dice(pred, label, self.num_classes)
            dice_scores.append(dice_result['dice_mean'])
            
            # ── Structural metrics (full eval only) ──
            if self.is_full_eval:
                prob = F.softmax(logits, dim=1)
                A_pred = soft_adjacency_from_probs(prob, self.model.connectivity)
                A_gt = hard_adjacency_from_mask(label, self.num_classes, self.model.connectivity)

                try:
                    struct = compute_structural_score(
                        pred, label, A_pred, A_gt, self.num_classes,
                        voxel_spacing=self.voxel_spacing,
                        compute_topology=self.eval_topology,
                        compute_surfaces=self.eval_surfaces,
                        topology_classes=topology_classes,
                        surface_classes=surface_classes,
                    )
                    for k, v in struct.items():
                        if isinstance(v, dict):
                            continue
                        val = v.item() if torch.is_tensor(v) else v
                        struct_accum[k] = struct_accum.get(k, 0.0) + val
                except Exception as e:
                    if n_batches == 0:
                        print(f"  ⚠ Structural metrics failed: {e}")

            n_batches += 1
        
        result = {k: v / max(n_batches, 1) for k, v in accum.items()}
        result['dice'] = sum(dice_scores) / max(len(dice_scores), 1)
        
        if self.is_full_eval:
            for k, v in struct_accum.items():
                result[k] = v / max(n_batches, 1)
        
        self.scheduler.step(result['total'])
        return result
    
    
    def get_lr(self) -> float:
        """현재 learning rate 반환"""
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'best_loss': self.best_loss,
            'lambda_hard': self.lambda_hard,
        }, path)
        
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_dice = checkpoint['best_dice']
        self.best_loss = checkpoint['best_loss']