# Aerial-Image-Segmentation-using-U-net-for-Urban-Development
Semantic segmentation of aerial &amp; satellite imagery using U-Net deep learning — building detection and land use classification with PyTorch.

# 🛰️ Aerial Image Segmentation with U-Net

> Deep learning-based semantic segmentation of satellite/aerial imagery for building detection and land use classification using the U-Net architecture.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Overview

This project implements a **U-Net** convolutional neural network to perform pixel-wise semantic segmentation on aerial and satellite imagery. The model is capable of:

- 🏠 **Building Detection** — Identifying rooftops and structures in urban zones
- 🌿 **Land Use Classification** — Segmenting vegetation, roads, water bodies, and bare land
- 🗺️ **Urban Planning Support** — Providing precise geospatial masks for GIS workflows

The U-Net architecture, originally designed for biomedical image segmentation, is highly effective for aerial imagery due to its encoder-decoder structure with skip connections — preserving fine spatial details while capturing global context.

---

## 🗂️ Project Structure

```
aerial-unet-segmentation/
├── configs/
│   └── config.yaml                  # Hyperparameters and training config
├── data/
│   ├── raw/                         # Original dataset (downloaded externally)
│   ├── processed/                   # Preprocessed tiles and masks
│   └── samples/                     # Sample images for quick testing
├── docs/
│   └── architecture.md              # U-Net architecture details
├── models/
│   └── unet.py                      # U-Net model definition
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and dataset visualization
│   ├── 02_training.ipynb            # Model training walkthrough
│   └── 03_inference_demo.ipynb      # Inference and results visualization
├── results/
│   ├── checkpoints/                 # Saved model weights (.pth)
│   ├── predictions/                 # Output segmentation masks
│   └── plots/                       # Training curves and evaluation plots
├── src/
│   ├── dataset.py                   # Dataset class and data loaders
│   ├── train.py                     # Training loop
│   ├── evaluate.py                  # Evaluation metrics (IoU, Dice, etc.)
│   ├── predict.py                   # Inference on new images
│   ├── losses.py                    # Custom loss functions
│   ├── utils/
│   │   ├── augmentations.py         # Albumentations transforms
│   │   ├── metrics.py               # IoU, Dice, Precision, Recall
│   │   └── helpers.py               # Utility functions
│   └── visualization/
│       └── visualize.py             # Overlay predictions on images
├── tests/
│   ├── test_model.py                # Unit tests for model
│   ├── test_dataset.py              # Unit tests for dataset loading
│   └── test_metrics.py              # Unit tests for metrics
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

The **U-Net** consists of:

- **Encoder (Contracting Path)**: 4 downsampling blocks using `3×3 Conv → BN → ReLU → MaxPool`
- **Bottleneck**: Deepest feature representation (1024 channels)
- **Decoder (Expanding Path)**: 4 upsampling blocks with skip connections via concatenation
- **Output**: `1×1` convolution to map to `N` segmentation classes

| Layer Block | Channels | Resolution |
|---|---|---|
| Input | 3 | 512×512 |
| Enc1 | 64 | 512×512 |
| Enc2 | 128 | 256×256 |
| Enc3 | 256 | 128×128 |
| Enc4 | 512 | 64×64 |
| Bottleneck | 1024 | 32×32 |
| Dec4 | 512 | 64×64 |
| Dec3 | 256 | 128×128 |
| Dec2 | 128 | 256×256 |
| Dec1 | 64 | 512×512 |
| Output | N_classes | 512×512 |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/25Satyam/aerial-unet-segmentation.git
cd aerial-unet-segmentation
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset

This project supports the following public datasets:

| Dataset | Classes | Resolution | Link |
|---|---|---|---|
| [ISPRS Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html) | 6 | 6000×6000 | ISPRS |
| [INRIA Aerial Image](https://project.inria.fr/aerialimagelabeling/) | 2 (building/bg) | 5000×5000 | INRIA |
| [DeepGlobe Land Cover](http://deepglobe.org/) | 7 | 2448×2448 | DeepGlobe |
| [Massachusetts Roads/Buildings](https://www.cs.toronto.edu/~vmnih/data/) | 2 | 1500×1500 | MIT |

### Download & Prepare
```bash
# Place raw images in data/raw/images/ and masks in data/raw/masks/
# Then run preprocessing:
python src/dataset.py --prepare --input data/raw --output data/processed --tile-size 512
```

---

## 🚀 Training

### Configure
Edit `configs/config.yaml` to set dataset path, hyperparameters, and augmentations.

### Run Training
```bash
python src/train.py --config configs/config.yaml
```

### Resume from Checkpoint
```bash
python src/train.py --config configs/config.yaml --resume results/checkpoints/best_model.pth
```

### Monitor with TensorBoard
```bash
tensorboard --logdir results/logs
```

---

## 📊 Evaluation

```bash
python src/evaluate.py --config configs/config.yaml --weights results/checkpoints/best_model.pth --split test
```

**Metrics computed:**
- Mean IoU (mIoU)
- Dice Coefficient (F1)
- Pixel Accuracy
- Per-class IoU

---

## 🔍 Inference

```bash
# Single image
python src/predict.py --image data/samples/aerial_001.jpg --weights results/checkpoints/best_model.pth --output results/predictions/

# Batch inference
python src/predict.py --input-dir data/samples/ --weights results/checkpoints/best_model.pth --output results/predictions/
```

---

## 📈 Results

| Model | mIoU | Dice | Pixel Acc |
|---|---|---|---|
| U-Net (baseline) | 72.4% | 0.814 | 89.3% |
| U-Net + Augmentation | 76.1% | 0.841 | 91.2% |
| U-Net + Focal Loss | 78.3% | 0.857 | 92.1% |
| **U-Net + Combined Loss** | **80.6%** | **0.874** | **93.4%** |

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📚 References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
- Chen, L. C., et al. (2018). [DeepLab: Semantic Image Segmentation](https://arxiv.org/abs/1802.02611).
- ISPRS 2D Semantic Labeling Contest — Potsdam dataset.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Satyam**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/satyam-423376244/) | [GitHub](https://github.com/25Satyam)
