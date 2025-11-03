# GLADE-Net Phase 1 (Grid UFormer)

Phase-1 training and inference pipeline for **GLADE-Net**, the first stage of our two-phase remote-sensing dehazing framework.  
This phase focuses on **global feature restoration** using grid-based partitioning and a UFormer backbone.

---

## 🧠 Overview

Phase 1 implements:
- Global attention encoder–decoder (UFormer) for coarse haze removal.
- PyTorch Lightning training loop with mixed-precision support.
- Grid dataset loader with on-the-fly tiling and augmentation.
- Gradual Frequency Loss (Charbonnier-based) with PSNR/SSIM metrics.
- Tools for dataset grid preparation and inference on large-scale aerial imagery.

Phase 2 will refine the results using a perceptual GAN in both RGB and Laplacian domains.

---

## 📂 Repository Structure

GLADE-Net-Phase1/
│
├── src/
│   ├── gladenet_phase1/
│   │   └── models/
│   │       └── uformer_module.py     # UFormer architecture
│   └── train.py                      # PyTorch-Lightning training entry
│
├── tools/
│   ├── grid_crop_pairs.py            # Pre-tile large training images
│   ├── grid_crop.py                  # Simple grid crop utility
│   └── inference_grid.py             # Phase-1 inference (new version)
│
├── dataset/
│   ├── train/
│   └── test/
│
├── requirements.txt
└── README.md

---

## ⚙️ Installation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

> Tested with Python 3.8+ and PyTorch ≥ 2.0.

---

## 🧩 Dataset Layout

Example folder structure:

/path/to/RICE_DATASET/
  ├── train/
  │   ├── cloud/   # hazy or cloudy input images
  │   └── label/   # clean ground-truth targets
  └── test/
      ├── cloud/
      └── label/

---

## 🧱 Step 1: Grid-Based Dataset Preparation

To generate 128×128 tiles for training and validation:

python tools/grid_crop_pairs.py \
  --root /media/anas/Data/Anas_Work_2/Grid_SPSR_paper/Grid_spsr_code/dataset \
  --out_root /media/anas/Data/Anas_Work_2/Grid_SPSR_paper/Grid_spsr_code/dataset_grid \
  --tile_w 128 --tile_h 128 --overlap 0

Output tiles will be created under:
dataset_grid/train/
dataset_grid/test/

---

## 🚀 Step 2: Training Phase 1 (UFormer)

Run training from the project root:

python -m src.train \
  --train_dir /media/anas/Data/Anas_Work_2/Grid_SPSR_paper/Grid_spsr_code/dataset_grid/train \
  --val_dir   /media/anas/Data/Anas_Work_2/Grid_SPSR_paper/Grid_spsr_code/dataset_grid/test \
  --img_size 128 \
  --batch_size 1 \
  --epochs 2500 \
  --devices 1 \
  --precision 16 \
  --accumulate 2 \
  --lr 2e-4 \
  --save_ckpt /media/anas/Data/Anas_Work_2/Grid_SPSR_paper/Grid_spsr_code/dataset_grid/Grid_RICE_1_new_dataset_final.ckpt

You can resume training with:

python -m src.train --train_dir ... --val_dir ... --resume /path/to/checkpoint.ckpt

---

## 🔍 Step 3: Inference Phase 1 (Grid UFormer)

Run inference on cloudy test images using the trained checkpoint:

python tools/inference_grid.py \
  --weights /media/anas/Data/Anas_Work_2/Grid_SPSR_paper/Grid_spsr_code/Grid/Grid_haze1k_freq_loss_train_with_test_full.ckpt \
  --input /media/anas/Data/Anas_Work_2/Grid_SPSR_paper/Grid_spsr_code/dataset/test/cloud \
  --output /media/anas/Data/Anas_Work_2/Grid_SPSR_paper/Grid_spsr_code/dataset/test/output_phase_1 \
  --tile 512 \
  --device cuda

The script will:
- Load each image, pad to a multiple of the tile size.
- Partition into non-overlapping grids.
- Apply the UFormer model patch-by-patch.
- Reconstruct and crop back to the original size.
- Save restored outputs under output_phase_1/.

Example output folders:
dataset/test/cloud/           # input hazy images
dataset/test/label/           # ground truth
dataset/test/output_phase_1/  # Phase 1 restored results

---

## 🧮 Step 4: Phase 2 – Perceptual GAN Refinement (Coming Soon)

The next phase will use the output from Phase 1 (restored dehazed images) paired with their ground-truth targets to train a perceptual enhancement GAN.  
This stage focuses on:
- Fine texture recovery and color correction.
- Multi-domain learning (RGB + Laplacian).
- Perceptual and adversarial loss integration.

### Placeholder for Phase 2 Code

#############################
# Phase 2 - Enhancement GAN
# (To be implemented)
#############################

Example usage (will be updated later):

python tools/inference_phase2.py \
  --weights /path/to/phase2_weights.ckpt \
  --input dataset/test/output_phase_1 \
  --gt dataset/test/label \
  --output dataset/test/output_phase_2

---

## 🧠 Citation

If you use this repository, please cite the paper:

Ali, A. M., Boulila, W., Benjdira, B., Ammar, A. et al.  
“CLEAR-Net: A Cascaded Local and External Attention Network for Enhanced Dehazing of Remote Sensing Images.”  
IEEE JSTARS, 2025.

---

## 📜 License

MIT License  
Copyright (c) 2025 Anas M. Ali
