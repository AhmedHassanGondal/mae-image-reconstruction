# Masked Autoencoder (MAE) for Image Reconstruction

This repository contains a PyTorch implementation of a Masked Autoencoder (MAE) for self-supervised learning on image data, specifically using the TinyImageNet dataset.

## Project Structure

```text
MAE-Image-Reconstruction/
├── notebooks/          # Jupyter notebooks with experiments
│   └── Assignment#2.ipynb
├── src/                # Source code (scripts for training and evaluation)
├── models/             # Saved model checkpoints
├── data/               # Dataset instructions/placeholders
├── results/            # Reconstruction results and plots
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

## Overview

Masked Autoencoders (MAE) are a scalable self-supervised learner for computer vision. As described in the original paper, MAEs mask random patches of the input image and reconstruct the missing pixels.

### Key Features
- **Vision Transformer (ViT) Encoder**: Processes only the visible patches.
- **Lightweight Decoder**: Reconstructs the full image from latent representations and mask tokens.
- **High Masking Ratio**: Uses a 75% masking ratio for effective self-supervised representation learning.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AhmedHassanGondal/MAE-Image-Reconstruction.git
   cd MAE-Image-Reconstruction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open the notebook in the `notebooks/` directory to see the full implementation, training loop, and reconstruction results.

```bash
jupyter notebook notebooks/Assignment#2.ipynb
```

## Results

The model achieves high-quality reconstruction of masked images by learning robust spatial representations. Sample results can be found in the notebook.

## License
MIT
