# AtlasGraphSeg

Anatomically-informed 3D brain segmentation with an **atlas-derived graph prior**.
The project fine-tunes a pretrained 3D segmentation network (nnU-Net) with a
differentiable ROI-adjacency loss built from a FreeSurfer/FastSurfer atlas, so that
predictions respect the expected anatomical neighborhood structure between regions.

## Idea

1. Build an **atlas graph prior** from `aparc+aseg.mgz`: remap FreeSurfer labels to the
   FastSurfer 78/79-class scheme, then estimate which regions are anatomically adjacent
   (`possible_mask`) and how strongly (`adjacency_strength`).
2. During fine-tuning, derive a **soft, differentiable ROI adjacency** from the network's
   per-voxel probabilities and match it to the atlas prior (penalizing anatomically
   impossible adjacencies).
3. Combine with segmentation and structural-regularization losses (e.g. island penalty)
   to reduce anatomically implausible predictions.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the loss-pipeline internals and conventions.

## Repository layout

```
atlasgraphseg/              # importable package
├── config.py               # paths & settings (read from environment variables)
├── data.py                 # FastSurfer label maps, dataset/dataloaders
├── graph.py                # soft/hard ROI adjacency from probs / masks
├── graph_prior.py          # connectivity offsets & tensor helpers
├── structural_prior.py     # loads the atlas prior (dispatch by num_classes)
├── metrics.py              # Dice, connected components, Betti, HD95/ASSD
├── node_edge.py            # standalone pure-PyTorch U-Net + prior loss (reference)
├── loss/                   # seg / hard-prior / island-penalty / orchestration
└── nnunet_integration/     # nnU-Net wrapper, graph-enhanced model, trainer, experiment
scripts/                    # command-line entry points
├── build_atlas_prior.py
├── run_experiment.py
└── visualize_atlas_prior.py
assets/                     # data & generated artifacts (tracked in git)
├── aparc+aseg.mgz          # FreeSurfer atlas volume (input to build_atlas_prior)
├── priors/                 # atlas_prior_{78,79}class.pt
└── figures/                # rendered figures
```

## Setup

```bash
pip install -e .            # installs the package + dependencies (editable)
```

Requires PyTorch (CUDA recommended), nnU-Net v2, and nibabel. An editable install is
needed so `scripts/*.py` can `import atlasgraphseg`.

### Environment variables

Environment-specific paths are **not** hardcoded — set them before running experiments:

| Variable | Meaning |
| --- | --- |
| `DATASET_ROOT` | nnU-Net raw dataset dir (must contain `imagesTr/`, `labelsTr/`) |
| `NNUNET_RESULTS` | Pretrained nnU-Net results dir or a checkpoint `.pth` (with sibling `plans.json` + `dataset.json`) |
| `ATLASGRAPHSEG_OUTPUT` | Where experiment outputs/checkpoints are written (default: `./experiments`) |
| `WANDB_PROJECT`, `WANDB_ENTITY` | Optional Weights & Biases logging |
| `ATLAS_PRIOR_PATH` | Override the default atlas prior (default: `assets/priors/atlas_prior_78class.pt`) |

Example:

```bash
export DATASET_ROOT=/data/nnUNet_raw/Dataset901_Oasis
export NNUNET_RESULTS=/data/nnUNet_results/Dataset901_Oasis/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
export ATLASGRAPHSEG_OUTPUT=/data/experiments
```

## Usage

Build the atlas prior from a FreeSurfer atlas volume:

```bash
python scripts/build_atlas_prior.py \
    --atlas assets/aparc+aseg.mgz \
    --output assets/priors/atlas_prior_78class.pt \
    --connectivity 6 --visualize
```

Run a fine-tuning experiment (evaluate pretrained baseline, then sweep graph-prior λ):

```bash
python scripts/run_experiment.py
```

Experiment knobs (batch size, patch size, λ values, epochs) live in
`atlasgraphseg/nnunet_integration/experiment.py::ExperimentConfig`.

Regenerate publication figures from a saved prior:

```bash
python scripts/visualize_atlas_prior.py --prior assets/priors/atlas_prior_79class.pt
```

## Notes

- Raw volumes (`*.mgz`, except the bundled atlas), Weights & Biases run logs (`wandb/`),
  and experiment outputs (`experiments/`) are excluded from version control.
