#!/usr/bin/env python
"""
효율적인 Pretrained + Graph Prior 실험

baseline (λ=0)은 이미 학습된 pretrained weights 그대로 평가만 하고,
graph prior (λ>0)만 fine-tuning합니다.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch

# Import from the installed package (pip install -e .)
from atlasgraphseg.nnunet_integration.experiment import (
    ExperimentConfig, setup_dataloaders, run_single_experiment,
)
from atlasgraphseg.nnunet_integration.nnunet_wrapper import nnUNetWrapper
from atlasgraphseg.nnunet_integration.graph_nnunet import GraphEnhancednnUNet, GraphPriorTrainer



def evaluate_pretrained_baseline(config, val_loader, pretrained_path):
    """
    Pretrained baseline을 그대로 평가만 (학습 X)
    
    Returns:
        baseline metrics (dice, structural score, etc.)
    """
    print("\n" + "="*70)
    print("📊 Baseline Evaluation (Pretrained nnU-Net, λ=0)")
    print("="*70)
    print("  → 이미 학습된 weights를 그대로 평가만 합니다 (학습 없음)")
    
    # Load pretrained model
    print(f"\n📦 Loading pretrained weights from {pretrained_path}")
    base_model = nnUNetWrapper(pretrained_path=pretrained_path, device=config.DEVICE)
    
    # Wrap with graph model (but λ=0 so no graph loss)
    model = GraphEnhancednnUNet(
        nnunet_model=base_model,
        num_classes=config.NUM_CLASSES,
        lambda_graph=0.0,  # No graph prior
        connectivity=config.CONNECTIVITY,
    )
    
    # Create trainer (just for validation)
    trainer = GraphPriorTrainer(
        model=model,
        train_loader=None,  # No training
        val_loader=val_loader,
        lr=config.LEARNING_RATE,
        device=config.DEVICE,
        num_classes=config.NUM_CLASSES,
        voxel_spacing=config.VOXEL_SPACING,
        eval_topology=config.EVAL_TOPOLOGY,
        eval_surfaces=config.EVAL_SURFACES,
    )

    trainer.is_full_eval = True  # Compute all metrics for baseline

    # Validate only
    print("\n🔍 Evaluating baseline on validation set...")
    baseline_metrics = trainer.validate()
    
    print("\n✅ Baseline Results:")
    print(f"  Dice:              {baseline_metrics['dice']:.4f}")
    if 'island_penalty' in baseline_metrics:
        print(f"  Island Penalty:    {baseline_metrics['island_penalty']:.4f}")
        print(f"  Adjacency F1:      {baseline_metrics['adj_f1']:.4f}")
        print(f"  Structural Score:  {baseline_metrics['structural_score']:.4f}")
    
    return baseline_metrics




def run_efficient_experiment():
    """
    효율적인 실험:
    1. Baseline은 평가만 (학습 X)
    2. Graph prior만 fine-tuning
    """
    print("="*70)
    print("🚀 Efficient Pretrained + Graph Prior Experiment")
    print("="*70)
    
    config = ExperimentConfig()
    
    # Check pretrained weights
    PRETRAINED_PATH = config.NNUNET_RESULTS
    
    if not Path(PRETRAINED_PATH).exists():
        print(f"\n❌ ERROR: Pretrained weights not found at {PRETRAINED_PATH}")
        return
    
    print(f"\n📋 실험 설정:")
    print(f"  - Pretrained weights: {PRETRAINED_PATH}")
    print(f"  - Dataset: {config.MRI_DIR}")
    print(f"  - Subset size: {config.SUBSET_SIZE}")
    print(f"  - Epochs: {config.NUM_EPOCHS}")
    print(f"  - Lambda values: {config.LAMBDA_GRAPH_VALUES[1:]}")  # Skip λ=0
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.OUTPUT_DIR = os.path.join(config.OUTPUT_ROOT, "graph_prior_experiments", timestamp)
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"  - 결과 저장: {config.OUTPUT_DIR}")
    
    # Load data
    print(f"\n📦 데이터 로딩 중...")
    train_loader, val_loader = setup_dataloaders(config)
    
    # ===== Step 1: Evaluate Baseline (no training) =====
    baseline_metrics = evaluate_pretrained_baseline(
        config, val_loader, PRETRAINED_PATH
    )
    
    # ===== Step 2: Train Graph Prior models =====
    results = []
    
    for lambda_val in config.LAMBDA_GRAPH_VALUES:  

        print(f"\n{'='*70}")
        print(f"🔬 Experiment: λ={lambda_val}")
        print(f"{'='*70}")
        
        result = run_single_experiment(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            lambda_graph=lambda_val,
            pretrained_path=PRETRAINED_PATH,
        )
        
        results.append(result)
    
    # ===== Step 3: Compare =====
    print("\n" + "="*80)
    print("📊 FINAL RESULTS COMPARISON")
    print("="*80)

    header = (f"{'Experiment':<35} {'λ':>6} {'Dice':>8} "
              f"{'HD95':>8} {'ASSD':>8} {'β-err':>8} "
              f"{'Island':>8} {'Adj F1':>8}")
    print(f"\n{header}")
    print("-" * len(header))

    # Baseline row
    print(f"{'Baseline (λ=0)':<35} {0.0:>6.2f} "
          f"{baseline_metrics['dice']:>8.4f} "
          f"{baseline_metrics.get('hd95', float('nan')):>8.2f} "
          f"{baseline_metrics.get('assd', float('nan')):>8.2f} "
          f"{baseline_metrics.get('betti_total_error', float('nan')):>8.2f} "
          f"{baseline_metrics.get('island_penalty', float('nan')):>8.2f} "
          f"{baseline_metrics.get('adj_f1', float('nan')):>8.4f}")

    # Graph prior rows
    for r in results:
        bvm = r.get('best_val_metrics', {})
        print(f"{r['experiment_name']:<35} {r['lambda_graph']:>6.2f} "
              f"{r['best_dice']:>8.4f} "
              f"{bvm.get('hd95', float('nan')):>8.2f} "
              f"{bvm.get('assd', float('nan')):>8.2f} "
              f"{bvm.get('betti_total_error', float('nan')):>8.2f} "
              f"{bvm.get('island_penalty', float('nan')):>8.2f} "
              f"{bvm.get('adj_f1', float('nan')):>8.4f}")

    # Improvement
    if results:
        print(f"\n{'Improvement vs Baseline':<35} {'λ':>6} {'ΔDice':>8} "
              f"{'ΔHD95':>8} {'ΔASSD':>8} {'Δβ-err':>8}")
        print("-" * 75)

        for r in results:
            bvm = r.get('best_val_metrics', {})
            print(f"{r['experiment_name']:<35} {r['lambda_graph']:>6.2f} "
                  f"{r['best_dice'] - baseline_metrics['dice']:>+8.4f} "
                  f"{bvm.get('hd95', 0) - baseline_metrics.get('hd95', 0):>+8.2f} "
                  f"{bvm.get('assd', 0) - baseline_metrics.get('assd', 0):>+8.2f} "
                  f"{bvm.get('betti_total_error', 0) - baseline_metrics.get('betti_total_error', 0):>+8.2f}")

    # Save
    results_summary = {
        'baseline': {'metrics': baseline_metrics, 'lambda': 0.0},
        'graph_prior_results': results,
        'config': {
            'num_epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'subset_size': config.SUBSET_SIZE,
            'lambda_values': config.LAMBDA_GRAPH_VALUES,
            'voxel_spacing': config.VOXEL_SPACING,
        }
    }

    results_path = Path(config.OUTPUT_DIR) / "results_summary.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\n✅ 실험 완료! 결과: {results_path}")
    return results_summary



if __name__ == "__main__":
    run_efficient_experiment()  