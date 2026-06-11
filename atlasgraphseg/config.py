"""
Centralized paths and configuration.

Environment-specific paths (dataset, pretrained nnU-Net checkpoint, output dir)
are read from environment variables so collaborators do not have to edit source.
Bundled assets (atlas priors, the FreeSurfer atlas volume) resolve to the repo's
``assets/`` directory by default.

Set the per-environment variables before running an experiment, e.g.::

    export DATASET_ROOT=/data/nnUNet_raw/Dataset901_Oasis
    export NNUNET_RESULTS=/data/nnUNet_results
    export ATLASGRAPHSEG_OUTPUT=/data/experiments
"""
import os
from pathlib import Path

# ── Repo / asset locations (resolved relative to this file) ──
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = Path(os.environ.get("ATLASGRAPHSEG_ASSETS", REPO_ROOT / "assets"))
PRIORS_DIR = ASSETS_DIR / "priors"
FIGURES_DIR = ASSETS_DIR / "figures"

# Default atlas prior files (produced by scripts/build_atlas_prior.py)
ATLAS_PRIOR_78 = Path(os.environ.get("ATLAS_PRIOR_PATH", PRIORS_DIR / "atlas_prior_78class.pt"))
ATLAS_PRIOR_79 = PRIORS_DIR / "atlas_prior_79class.pt"

# Default FreeSurfer atlas volume (input to build_atlas_prior.py)
ATLAS_VOLUME = ASSETS_DIR / "aparc+aseg.mgz"

# ── Per-environment paths (override via environment variables) ──
# Root of the nnU-Net raw dataset (must contain imagesTr/ and labelsTr/).
DATASET_ROOT = os.environ.get("DATASET_ROOT", "")

# nnU-Net results dir or a specific checkpoint .pth.
NNUNET_RESULTS = os.environ.get("NNUNET_RESULTS", "")

# Where experiment outputs / checkpoints are written.
OUTPUT_ROOT = os.environ.get("ATLASGRAPHSEG_OUTPUT", str(REPO_ROOT / "experiments"))

# Weights & Biases (optional).
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "GraphSeg_nnunet_comparison")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
