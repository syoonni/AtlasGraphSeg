# Architecture

This document describes the project's structure, common commands, and codebase-specific conventions.

## What this project does

AtlasGraphSeg fine-tunes a **pretrained nnU-Net** for 3D brain segmentation (FastSurfer 79-class scheme: 78 foreground + background) by adding an **atlas-derived graph-prior loss**. The core idea: derive a region-adjacency graph from a FreeSurfer atlas, then penalize the network for predicting anatomically *impossible* region adjacencies. The backbone weights and architecture come from an existing nnU-Net training run — this repo never trains a segmentation model from scratch.

## Commands

```bash
pip install -e .                         # editable install; needed so scripts can import atlasgraphseg

# 1. Build the atlas graph prior from a FreeSurfer atlas volume
python scripts/build_atlas_prior.py \
    --atlas assets/aparc+aseg.mgz \
    --output assets/priors/atlas_prior_78class.pt \
    --connectivity 6 --min-contact 5 --visualize

# 2. Run the fine-tuning experiment (baseline eval + graph-prior sweep)
python scripts/run_experiment.py

# 3. Regenerate publication figures from a saved prior
python scripts/visualize_atlas_prior.py --prior assets/priors/atlas_prior_79class.pt
```

There is **no test suite or linter**. "Running" the project means executing `scripts/run_experiment.py`, which requires a GPU, a trained nnU-Net checkpoint, and the OASIS dataset — supplied via environment variables (see Environment configuration below).

## Architecture: the loss pipeline is the heart of the project

Everything centers on converting a 3D segmentation into a **region-adjacency matrix** `A` of shape `[B, C, C]` where `C = num_classes - 1` (background channel 0 is always excluded), then comparing predicted vs. atlas-expected adjacency.

Data flow for a single training step (`GraphPriorTrainer.train_epoch` in `atlasgraphseg/nnunet_integration/graph_nnunet.py`):

1. `nnUNetWrapper` (`atlasgraphseg/nnunet_integration/nnunet_wrapper.py`) reconstructs the nnU-Net network from `plans.json`/`dataset.json` next to the checkpoint and loads weights. Its `forward` discards deep-supervision outputs, returning only full-res logits.
2. `GraphEnhancednnUNet.compute_loss` → `atlasgraphseg/loss/total_loss.py::compute_all_losses` returns **three independent, unweighted scalars**: `seg` (Dice+CE), `hard` (atlas-prior penalty), `island` (disconnected-component penalty, eval-only).
3. The trainer sums them with external lambda weights: `seg + lambda_hard * hard + lambda_island * island`. **`lambda_graph` from the config maps to `lambda_hard`** — that is the knob the experiments sweep.

Key conversion functions in `atlasgraphseg/graph.py`:
- `soft_adjacency_from_probs` — differentiable adjacency from softmax probabilities (used for the prediction; this is what carries gradient).
- `hard_adjacency_from_mask` — adjacency from integer ground-truth labels.
Both use neighborhood offsets from `atlasgraphseg/graph_prior.py::connectivity_offsets` (`'6-connectivity'` / `'18-'` / `'26-'`) and an einsum over shifted volumes. They are written to produce **matching `[B, C, C]` shapes** — when editing one, keep the other's output dimension in lockstep.

`atlasgraphseg/loss/hard_loss.py::HardPriorLoss` only penalizes adjacencies on edges the atlas marks **impossible** (`impossible_mask = 1 - possible_mask`), after Frobenius-normalizing the predicted adjacency. Positive/connectivity enforcement is off by default.

## How the atlas prior is produced and consumed

- **Producer:** `scripts/build_atlas_prior.py` remaps FreeSurfer labels → FastSurfer 78-class via the `FREESURFER_TO_FASTSURFER` dict, computes voxel-level adjacency, and saves a dict (`possible_mask`, `adjacency_strength`, `adjacency_binary`, `raw_contact_counts`, `config`, ...) as a `.pt` file. Committed priors live in `assets/priors/atlas_prior_{78,79}class.pt`.
- **Consumer:** `atlasgraphseg/structural_prior.py::get_priors_for_num_classes(num_classes)` is the single dispatch point. For `num_classes == 79` it loads the atlas `.pt`, and **falls back to a hardcoded manual prior if the file is missing** (so a missing prior degrades silently rather than crashing). 5-class and other counts have their own hand-built priors. `HardPriorLoss` calls this at init.

## Conventions specific to this codebase

- **Class indexing:** labels are `0..num_classes-1` with `0 = background`. All adjacency matrices drop background, so a `[C, C]` matrix indexes foreground classes shifted down by one. Mixing these conventions is the most likely source of bugs.
- **Island penalty is eval-only:** `compute_all_losses` returns `island = 0` whenever `is_train=True` (guarded by the `is_train` flag), so it never affects gradients despite being summed in.
- **Expensive metrics are gated:** Betti-number topology and HD95/ASSD surface metrics run only on `EVAL_FULL_EPOCH` boundaries and only over `GraphPriorTrainer.KEY_STRUCTURES` (a clinically-selected subset), not all 78 classes. `trainer.is_full_eval` / `eval_topology` / `eval_surfaces` toggle this per epoch.
- **AMP:** training uses autocast + GradScaler, but adjacency/hard-loss computation is forced to FP32 (`autocast(enabled=False)`) for numerical stability. NaN/Inf losses are skipped, not crashed on.
- **Experiment knobs live in one class:** `atlasgraphseg/nnunet_integration/experiment.py::ExperimentConfig` (batch size, patch size, lambdas, eval flags). There is no CLI/argparse for experiments — edit this class to change runs. Environment-specific *paths* are not here; they come from `atlasgraphseg/config.py`.

## Environment configuration

`atlasgraphseg/config.py` is the single place paths resolve. **Bundled assets** (atlas priors, the `aparc+aseg.mgz` volume) default to the repo's `assets/` dir. **Per-environment paths** are read from environment variables and must be set before running:

- `DATASET_ROOT` — nnU-Net raw dataset (`imagesTr/`, `labelsTr/`).
- `NNUNET_RESULTS` — pretrained checkpoint `.pth` (or its results dir) with sibling `plans.json` + `dataset.json`. The pipeline raises `FileNotFoundError` early if absent.
- `ATLASGRAPHSEG_OUTPUT` — output/checkpoint dir (default `./experiments`).
- `WANDB_PROJECT` / `WANDB_ENTITY` — optional logging; empty entity falls back to the W&B default.
- `ATLAS_PRIOR_PATH` — override the default atlas prior file.

## Excluded from version control

`.gitignore` drops `wandb/` (run logs), `__pycache__/`, `experiments/`, build artifacts, and raw `*.mgz` volumes **except** the bundled `assets/aparc+aseg.mgz`. Generated `.pt` priors and figure assets under `assets/` *are* committed.
