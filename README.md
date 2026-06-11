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
   per-voxel probabilities and match it to the atlas prior.
3. Combine with segmentation and structural-regularization losses (e.g. island penalty)
   to reduce anatomically implausible predictions.

## Repository layout

| Path | Description |
| --- | --- |
| `build_atlas_prior.py` | Build the atlas graph prior (`possible_mask`, `adjacency_strength`) from `aparc+aseg.mgz`. |
| `graph_prior.py` | Connectivity offsets / graph-prior utilities. |
| `structural_prior.py` | Loads and applies the precomputed atlas prior. |
| `node_edge.py` | Pure-PyTorch 3D U-Net + differentiable soft ROI-adjacency prior loss. |
| `graph.py` | Graph construction helpers. |
| `data.py` | FastSurfer label maps and dataset/dataloader utilities. |
| `metrics.py` | Structural quality metrics (connected components, island penalty, etc.). |
| `loss/` | Segmentation, hard, island-penalty and combined losses. |
| `nnunet_integration/` | nnU-Net wrapper, graph-enhanced model and experiment runner. |
| `run_experiment.py` | Entry point: evaluate pretrained baseline and fine-tune with the graph prior. |
| `visualize_atlas_prior.py` | Render publication figures of the atlas prior. |
| `atlas_prior_78class.pt` / `atlas_prior_79class.pt` | Precomputed atlas graph priors. |

## Setup

```bash
pip install -r requirements.txt
```

Requires PyTorch (CUDA recommended), nnU-Net v2, and nibabel.

## Usage

Build the atlas prior from a FreeSurfer atlas volume:

```bash
python build_atlas_prior.py \
    --atlas /path/to/aparc+aseg.mgz \
    --output atlas_prior_78class.pt \
    --connectivity 6 --visualize
```

Run a fine-tuning experiment:

```bash
python run_experiment.py
```

> Note: some scripts contain hardcoded absolute paths (e.g. dataset / pretrained-weight
> locations). Adjust them for your environment before running.

## Notes

- Raw volumes (`*.mgz`) and Weights & Biases run logs (`wandb/`) are intentionally
  excluded from version control via `.gitignore`.
