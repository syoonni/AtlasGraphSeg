# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

AtlasGraphSeg fine-tunes a **pretrained nnU-Net** for 3D brain segmentation (FastSurfer 79-class scheme: 78 foreground + background) by adding an **atlas-derived graph-prior loss**. The core idea: derive a region-adjacency graph from a FreeSurfer atlas, then penalize the network for predicting anatomically *impossible* region adjacencies. The backbone weights and architecture come from an existing nnU-Net training run — this repo never trains a segmentation model from scratch.

## Commands

```bash
pip install -r requirements.txt          # torch, nnunetv2, nibabel, etc.

# 1. Build the atlas graph prior from a FreeSurfer atlas volume
python build_atlas_prior.py \
    --atlas /path/to/aparc+aseg.mgz \
    --output atlas_prior_78class.pt \
    --connectivity 6 --min-contact 5 --visualize

# 2. Run the fine-tuning experiment (baseline eval + graph-prior sweep)
python run_experiment.py

# 3. Regenerate publication figures from a saved prior
python visualize_atlas_prior.py --prior atlas_prior_79class.pt
```

There is **no test suite, linter, or build step**. "Running" the project means executing `run_experiment.py`, which requires GPU, a trained nnU-Net checkpoint, and the OASIS dataset (see Hardcoded paths below).

## Architecture: the loss pipeline is the heart of the project

Everything centers on converting a 3D segmentation into a **region-adjacency matrix** `A` of shape `[B, C, C]` where `C = num_classes - 1` (background channel 0 is always excluded), then comparing predicted vs. atlas-expected adjacency.

Data flow for a single training step (`GraphPriorTrainer.train_epoch` in `nnunet_integration/graph_nnunet.py`):

1. `nnUNetWrapper` (`nnunet_integration/nnunet_wrapper.py`) reconstructs the nnU-Net network from `plans.json`/`dataset.json` next to the checkpoint and loads weights. Its `forward` discards deep-supervision outputs, returning only full-res logits.
2. `GraphEnhancednnUNet.compute_loss` → `loss/total_loss.py::compute_all_losses` returns **three independent, unweighted scalars**: `seg` (Dice+CE), `hard` (atlas-prior penalty), `island` (disconnected-component penalty, eval-only).
3. The trainer sums them with external lambda weights: `seg + lambda_hard * hard + lambda_island * island`. **`lambda_graph` from the config maps to `lambda_hard`** — that is the knob the experiments sweep.

Key conversion functions in `graph.py`:
- `soft_adjacency_from_probs` — differentiable adjacency from softmax probabilities (used for the prediction; this is what carries gradient).
- `hard_adjacency_from_mask` — adjacency from integer ground-truth labels.
Both use neighborhood offsets from `graph_prior.py::connectivity_offsets` (`'6-connectivity'` / `'18-'` / `'26-'`) and an einsum over shifted volumes. They are written to produce **matching `[B, C, C]` shapes** — when editing one, keep the other's output dimension in lockstep.

`loss/hard_loss.py::HardPriorLoss` only penalizes adjacencies on edges the atlas marks **impossible** (`impossible_mask = 1 - possible_mask`), after Frobenius-normalizing the predicted adjacency. Positive/connectivity enforcement is off by default.

## How the atlas prior is produced and consumed

- **Producer:** `build_atlas_prior.py` remaps FreeSurfer labels → FastSurfer 78-class via the `FREESURFER_TO_FASTSURFER` dict, computes voxel-level adjacency, and saves a dict (`possible_mask`, `adjacency_strength`, `adjacency_binary`, `raw_contact_counts`, `config`, ...) as a `.pt` file. Committed priors: `atlas_prior_78class.pt`, `atlas_prior_79class.pt`.
- **Consumer:** `structural_prior.py::get_priors_for_num_classes(num_classes)` is the single dispatch point. For `num_classes == 79` it loads the atlas `.pt`, and **falls back to a hardcoded manual prior if the file is missing** (so a missing prior degrades silently rather than crashing). 5-class and other counts have their own hand-built priors. `HardPriorLoss` calls this at init.

## Conventions specific to this codebase

- **Class indexing:** labels are `0..num_classes-1` with `0 = background`. All adjacency matrices drop background, so a `[C, C]` matrix indexes foreground classes shifted down by one. Mixing these conventions is the most likely source of bugs.
- **Island penalty is eval-only:** `compute_all_losses` returns `island = 0` whenever `is_train=True` (guarded by the `is_train` flag), so it never affects gradients despite being summed in.
- **Expensive metrics are gated:** Betti-number topology and HD95/ASSD surface metrics run only on `EVAL_FULL_EPOCH` boundaries and only over `GraphPriorTrainer.KEY_STRUCTURES` (a clinically-selected subset), not all 78 classes. `trainer.is_full_eval` / `eval_topology` / `eval_surfaces` toggle this per epoch.
- **AMP:** training uses autocast + GradScaler, but adjacency/hard-loss computation is forced to FP32 (`autocast(enabled=False)`) for numerical stability. NaN/Inf losses are skipped, not crashed on.
- **All config lives in one class:** `nnunet_integration/experiment.py::ExperimentConfig` (paths, batch size, patch size, lambdas, W&B project/entity). There is no CLI/argparse for experiments — edit this class to change runs.

## Hardcoded paths (must edit before running elsewhere)

These are absolute paths to the original author's environment and **will not exist** on another machine:
- `run_experiment.py` — `sys.path.append('/home/hwlim/syoon/AtlasGraphSeg')` and output dir `/home/hwlim/syoon/save/...`.
- `ExperimentConfig` — `DATA_ROOT` (OASIS Dataset901), `NNUNET_PATH`/`NNUNET_RESULTS` (pretrained checkpoint), and W&B `WANDB_ENTITY`.
The pipeline hard-requires a valid pretrained nnU-Net checkpoint with sibling `plans.json` + `dataset.json`; it raises `FileNotFoundError` early if absent.

## Excluded from version control

`.gitignore` drops `wandb/` (run logs), `__pycache__/`, and raw `*.mgz` volumes. Generated `.pt` priors and figure assets *are* committed.
