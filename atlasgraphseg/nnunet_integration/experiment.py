"""
nnU-Net Baseline vs nnU-Net + Graph Prior 비교 실험
"""
import wandb
from tqdm import tqdm
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader

from .nnunet_wrapper import nnUNetWrapper
from .graph_nnunet import GraphEnhancednnUNet, GraphPriorTrainer
from atlasgraphseg.data import FreeSurferDataset
from atlasgraphseg import config


class ExperimentConfig:
    """실험 설정. 환경 의존 경로는 atlasgraphseg.config(환경변수)에서 읽음."""
    # 데이터 경로 (DATASET_ROOT 환경변수로 지정)
    DATA_ROOT = config.DATASET_ROOT
    MRI_DIR = os.path.join(DATA_ROOT, "imagesTr") if DATA_ROOT else ""
    SEG_DIR = os.path.join(DATA_ROOT, "labelsTr") if DATA_ROOT else ""

    # nnU-Net pretrained 체크포인트 (NNUNET_RESULTS 환경변수로 지정)
    NNUNET_RESULTS = config.NNUNET_RESULTS

    # 학습 설정
    BATCH_SIZE = 4
    PATCH_SIZE = (128, 128, 112)  # (112, 128, 128)
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-5

    # 데이터셋 서브셋 (빠른 실험용)
    SUBSET_SIZE = None
    
    # 모델 설정
    NUM_CLASSES = 79  # 78 classes + 1 background (FastSurfer 기준)  
    
    # Graph prior 설정
    LAMBDA_GRAPH_VALUES = [2] # 0 (baseline), 0.1, 1.0, 2.0
    CONNECTIVITY = '6-connectivity'

    # Combined Graph Loss
    LAMBDA_ISLAND = 0.02

    # Voxel spacing (from ADNI/OASIS typical 1mm isotropic)
    VOXEL_SPACING = (1.0, 1.0, 1.0)
    
    # Evaluation settings
    EVAL_TOPOLOGY = True      # Betti numbers (slow but important for paper)
    EVAL_SURFACES = True      # HD95/ASSD (slow but essential for paper)
    EVAL_FULL_EPOCH = 10       # Compute expensive metrics every N epochs
    
    # 출력 (ATLASGRAPHSEG_OUTPUT 환경변수로 지정)
    OUTPUT_ROOT = config.OUTPUT_ROOT
    OUTPUT_DIR = config.OUTPUT_ROOT
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # WandB (WANDB_PROJECT / WANDB_ENTITY 환경변수로 지정)
    WANDB_PROJECT = config.WANDB_PROJECT
    WANDB_ENTITY = config.WANDB_ENTITY


def setup_dataloaders(config: ExperimentConfig):
    """데이터 로더 설정 (300개 Subset 적용)"""
    dataset = FreeSurferDataset(
        mri_dir=config.MRI_DIR,
        seg_dir=config.SEG_DIR,
        patch_size=config.PATCH_SIZE,
        use_simplified_labels=False,
        augment=True,
    )
    
    n_total = len(dataset)
    
    # [수정됨] 전체 데이터 vs Subset 분할 로직
    if config.SUBSET_SIZE and config.SUBSET_SIZE < n_total:
        target_len = config.SUBSET_SIZE
        print(f"✂️ Cutting dataset: Using {target_len} out of {n_total} samples")
        
        # 300개 안에서 8:2 나누기
        n_train = int(target_len * 0.8)  # 240
        n_val = target_len - n_train     # 60
        n_rest = n_total - target_len    # 나머지 버림
        
        # [Train, Val, 버릴것] 3분할
        train_set, val_set, _ = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_rest],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        # 기존 로직 (전체 사용)
        print(f"Using Full Dataset: {n_total} samples")
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
    
    # [최적화] persistent_workers=True 추가 (속도 향상)
    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=12, pin_memory=True,
        persistent_workers=True 
    )
    val_loader = DataLoader(
        val_set, batch_size=1,
        shuffle=False, num_workers=4,
        persistent_workers=True
    )
    
    print(f"Dataset Loaded: Train({n_train}), Val({n_val})")
    return train_loader, val_loader


def run_single_experiment(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    lambda_graph: float,
    pretrained_path: Optional[str] = None,
) -> Dict:
    """
    단일 실험 실행.
    
    Args:
        lambda_graph: 0이면 baseline, >0이면 graph prior 사용
        use_pretrained: nnU-Net pretrained weights 사용 여부
        pretrained_path: pretrained model 경로
    """
    # Validate pretrained path
    if not pretrained_path or not Path(pretrained_path).exists():
        raise FileNotFoundError(
            f"Pretrained weights required: {pretrained_path}\n"
            f"Please check the path exists."
        )
    
    experiment_name = f"lambda_{lambda_graph:.2f}_pretrained"

    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY or None,
        name=experiment_name,
        config={
            "lambda_graph": lambda_graph,
            "num_classes": config.NUM_CLASSES,  # NEW
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "patch_size": config.PATCH_SIZE,
            "voxel_spacing": config.VOXEL_SPACING,
            "eval_topology": config.EVAL_TOPOLOGY,
            "eval_surfaces": config.EVAL_SURFACES,
        },
        reinit=True,
        group="pretrained_graph"  # SIMPLIFIED: one group
    )
    
    print(f"\n{'='*60}")
    print(f"🔬 Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Load pretrained nnU-Net (REQUIRED)
    print(f"\n📦 Loading pretrained nnU-Net...")
    try:
        base_model = nnUNetWrapper(
            pretrained_path=pretrained_path,
            device=config.DEVICE
        )
        print(f"  ✓ Loaded from {pretrained_path}")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        raise RuntimeError(f"Cannot proceed without pretrained weights: {e}")
    
    # Graph-enhanced 모델로 래핑
    print(f"\n🔧 Creating graph-enhanced model...")
    model = GraphEnhancednnUNet(
        nnunet_model=base_model,
        num_classes=config.NUM_CLASSES,
        lambda_graph=lambda_graph,
        connectivity=config.CONNECTIVITY,
        possible_mask=None,  # Auto-load in loss.py
        expected_strength=None,  # Auto-load in loss.py
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  - Parameters: {n_params:,}")
    
    # Trainer 생성
    trainer = GraphPriorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config.LEARNING_RATE,
        device=config.DEVICE,
        num_classes=config.NUM_CLASSES,  # Pass num_classes
        voxel_spacing=config.VOXEL_SPACING,
        eval_topology=config.EVAL_TOPOLOGY,
        eval_surfaces=config.EVAL_SURFACES,
    )
    
    # 학습
    print(f"\n🚀 Starting training for {config.NUM_EPOCHS} epochs...")
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_dice = 0
    best_epoch = 0
    best_val_metrics = {}
    last_val_metrics = {}

    epoch_bar = tqdm(range(1, config.NUM_EPOCHS + 1), desc="Training Progress")
    
    for epoch in epoch_bar:
        
        # 비싼 메트릭은 N epoch마다만
        is_full_eval = (epoch % config.EVAL_FULL_EPOCH == 0) or (epoch == config.NUM_EPOCHS)
        trainer.is_full_eval = is_full_eval
        trainer.eval_topology = config.EVAL_TOPOLOGY and is_full_eval
        trainer.eval_surfaces = config.EVAL_SURFACES and is_full_eval 
        
        # Train
        train_metrics = trainer.train_epoch()
        
        # Validate
        val_metrics = trainer.validate()
        last_val_metrics = val_metrics.copy() 
        
        # History
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['val_dice'].append(val_metrics['dice'])

        # WandB logging
        log_dict = {
            "epoch": epoch,
            "train_seg_loss": train_metrics.get('seg', 0),
            "train_hard_loss": train_metrics.get('hard', 0),
            "train_loss": train_metrics.get('total', 0),
            "val_loss": val_metrics['total'],
            "val_dice": val_metrics['dice'],
        }
        
        # Structural metrics (only when lambda_graph > 0)
        if lambda_graph > 0:
            log_dict.update({
                "val_island_penalty": val_metrics.get('island_penalty', 0),
                "val_adj_f1": val_metrics.get('adj_f1', 0),
                "val_structural_score": val_metrics.get('structural_score', 0),
            })

        # Topology metrics (when computed)
        if 'betti_0_error' in val_metrics:
            log_dict.update({
                "val_betti_0_error": val_metrics['betti_0_error'],
                "val_betti_1_error": val_metrics['betti_1_error'],
                "val_betti_2_error": val_metrics['betti_2_error'],
                "val_betti_total_error": val_metrics['betti_total_error'],
            })
        
        # Surface distance metrics (when computed)
        if 'hd95' in val_metrics:
            log_dict.update({
                "val_hd95": val_metrics['hd95'],
                "val_assd": val_metrics['assd'],
            })
        
        wandb.log(log_dict)
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            best_val_metrics = val_metrics.copy()

            if 'structural_score' not in best_val_metrics:
                trainer.is_full_eval = True
                trainer.eval_topology = config.EVAL_TOPOLOGY
                trainer.eval_surfaces = config.EVAL_SURFACES
                full_metrics = trainer.validate()
                best_val_metrics.update({k: v for k, v in full_metrics.items() if k not in best_val_metrics})

                trainer.is_full_eval = is_full_eval
                trainer.eval_topology = config.EVAL_TOPOLOGY and is_full_eval
                trainer.eval_surfaces = config.EVAL_SURFACES and is_full_eval
            
            save_path = Path(config.OUTPUT_DIR) / experiment_name / "best_model.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

            wandb.run.summary["best_dice"] = best_dice
            wandb.run.summary["best_epoch"] = best_epoch

            # Log best structural metrics
            for key in ['hd95', 'assd', 'betti_0_error', 'betti_total_error',
                        'island_penalty', 'adj_f1', 'structural_score']:
                if key in val_metrics:
                    wandb.run.summary[f"best_{key}"] = val_metrics[key]

        # Save last model (every epoch, overwrite)
        last_save_path = Path(config.OUTPUT_DIR) / experiment_name / "last_model.pth"
        last_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), last_save_path)
        
        # Progress bar
        postfix = {
            'loss': f"{train_metrics['total']:.3f}",
            'dice': f"{val_metrics['dice']:.3f}",
            'best': f"{best_dice:.3f}",
        }
        if 'hd95' in val_metrics and val_metrics['hd95'] < float('inf'):
            postfix['hd95'] = f"{val_metrics['hd95']:.1f}"
        epoch_bar.set_postfix(postfix)
    
    wandb.finish()

    if 'structural_score' not in best_val_metrics:
        # best checkpoint 로드 후 full eval
        best_ckpt = Path(config.OUTPUT_DIR) / experiment_name / "best_model.pth"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=config.DEVICE))
        trainer.is_full_eval = True
        trainer.eval_topology = config.EVAL_TOPOLOGY
        trainer.eval_surfaces = config.EVAL_SURFACES
        full_metrics = trainer.validate()
        best_val_metrics.update({k: v for k, v in full_metrics.items()
                                if k not in best_val_metrics})

    if 'structural_score' not in last_val_metrics:
        # last checkpoint 로드 후 full eval
        last_ckpt = Path(config.OUTPUT_DIR) / experiment_name / "last_model.pth"
        if last_ckpt.exists():
            model.load_state_dict(torch.load(last_ckpt, map_location=config.DEVICE))
        trainer.is_full_eval = True
        trainer.eval_topology = config.EVAL_TOPOLOGY
        trainer.eval_surfaces = config.EVAL_SURFACES
        last_val_metrics = trainer.validate()

    print(f"\n✔ Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    if 'hd95' in best_val_metrics:
        print(f"  HD95: {best_val_metrics.get('hd95', 'N/A'):.2f} mm")
        print(f"  ASSD: {best_val_metrics.get('assd', 'N/A'):.2f} mm")
    if 'betti_total_error' in best_val_metrics:
        print(f"  Betti Total Error: {best_val_metrics.get('betti_total_error', 'N/A'):.2f}")
    
    return {
        'experiment_name': experiment_name,
        'lambda_graph': lambda_graph,
        'best_dice': best_dice,
        'best_epoch': best_epoch,
        'best_val_metrics': best_val_metrics,
        'last_val_metrics': last_val_metrics,
        'history': history,
    }


def run_comparison_experiment(config: ExperimentConfig):
    """
    전체 비교 실험 실행.
    """
    print("="*60)
    print("🧠 nnU-Net vs nnU-Net + Graph Prior Comparison")
    print("="*60)
    
    if not Path(config.NNUNET_RESULTS).exists():
        raise FileNotFoundError(
            f"❌ Pretrained weights not found: {config.NNUNET_RESULTS}\n"
            f"This experiment requires pretrained nnU-Net weights.\n"
            f"Please train nnU-Net first or check the path."
        )
    
    print(f"\n✔ Found pretrained weights: {config.NNUNET_RESULTS}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.OUTPUT_DIR = f"./experiments/{timestamp}"
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"✔ Output directory: {config.OUTPUT_DIR}")
    
    print(f"\n📦 Loading data...")
    train_loader, val_loader = setup_dataloaders(config)
    
    print(f"\n⚙️  Experiment Configuration:")
    print(f"  - Dataset: {config.MRI_DIR}")
    print(f"  - Num classes: {config.NUM_CLASSES}")
    print(f"  - Subset size: {config.SUBSET_SIZE if config.SUBSET_SIZE else 'Full'}")
    print(f"  - Epochs: {config.NUM_EPOCHS}")
    print(f"  - Lambda values: {config.LAMBDA_GRAPH_VALUES}")
    print(f"  - Voxel spacing: {config.VOXEL_SPACING}")
    print(f"  - Eval topology (Betti): {config.EVAL_TOPOLOGY}")
    print(f"  - Eval surfaces (HD95/ASSD): {config.EVAL_SURFACES}")
    
    # Run experiments
    results = []
    
    print("\n" + "="*60)
    print("🚀 Running Experiments")
    print("="*60)
    
    for i, lambda_val in enumerate(config.LAMBDA_GRAPH_VALUES, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(config.LAMBDA_GRAPH_VALUES)}: λ_graph={lambda_val}")
        print(f"{'='*60}")
        
        try:
            result = run_single_experiment(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                lambda_graph=lambda_val,
                pretrained_path=config.NNUNET_RESULTS,
            )
            results.append(result)
            
            print(f"\n✔ Completed: {result['experiment_name']}")
            print(f"  Best Dice: {result['best_dice']:.4f} (epoch {result['best_epoch']})")
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ===== Results Summary =====
    if not results:
        print("\n❌ No experiments completed successfully!")
        return []
    
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    # Header
    header = (f"{'Rank':<6} {'Experiment':<30} {'λ':>6} {'Dice':>8} "
              f"{'HD95':>8} {'ASSD':>8} {'β-err':>8} "
              f"{'Island':>8} {'Adj F1':>8} {'Struct':>8}")
    print(f"\n{header}")
    print("-" * len(header))
    
    results_sorted = sorted(results, key=lambda x: x['best_dice'], reverse=True)
    
    for rank, r in enumerate(results_sorted, 1):
        bvm = r.get('best_val_metrics', {})
        marker = "🏆" if rank == 1 else "  "
        print(f"{marker} {rank:<3} {r['experiment_name']:<30} {r['lambda_graph']:>6.2f} "
              f"{r['best_dice']:>8.4f} "
              f"{bvm.get('hd95', float('nan')):>8.2f} "
              f"{bvm.get('assd', float('nan')):>8.2f} "
              f"{bvm.get('betti_total_error', float('nan')):>8.2f} "
              f"{bvm.get('island_penalty', float('nan')):>8.2f} "
              f"{bvm.get('adj_f1', float('nan')):>8.4f} "
              f"{bvm.get('structural_score', float('nan')):>8.4f}")
    
    # Improvement table
    baseline = next((r for r in results if r['lambda_graph'] == 0.0), None)
    
    if baseline and len(results) > 1:
        print(f"\n{'Improvement vs Baseline':<30} {'λ':>6} {'ΔDice':>8} "
              f"{'ΔHD95':>8} {'ΔASSD':>8} {'Δβ-err':>8}")
        print("-"*75)
        
        b_bvm = baseline.get('best_val_metrics', {})
        
        for r in results:
            if r['lambda_graph'] == 0.0:
                continue
            bvm = r.get('best_val_metrics', {})
            
            d_dice = r['best_dice'] - baseline['best_dice']
            d_hd95 = bvm.get('hd95', 0) - b_bvm.get('hd95', 0)
            d_assd = bvm.get('assd', 0) - b_bvm.get('assd', 0)
            d_betti = bvm.get('betti_total_error', 0) - b_bvm.get('betti_total_error', 0)
            
            print(f"{r['experiment_name']:<30} {r['lambda_graph']:>6.2f} "
                  f"{d_dice:>+8.4f} "
                  f"{d_hd95:>+8.2f} "
                  f"{d_assd:>+8.2f} "
                  f"{d_betti:>+8.2f}")
    
    # Save results
    results_path = Path(config.OUTPUT_DIR) / "results.json"
    with open(results_path, 'w') as f:
        results_data = {
            'config': {
                'num_classes': config.NUM_CLASSES,
                'subset_size': config.SUBSET_SIZE,
                'num_epochs': config.NUM_EPOCHS,
                'lambda_values': config.LAMBDA_GRAPH_VALUES,
                'pretrained_path': str(config.NNUNET_RESULTS),
                'voxel_spacing': config.VOXEL_SPACING,
                'eval_topology': config.EVAL_TOPOLOGY,
                'eval_surfaces': config.EVAL_SURFACES,
            },
            'results': results,
        }
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {results_path}")
    
    return results