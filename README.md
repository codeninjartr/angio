# Angio: Blood Vessel Segmentation & Analysis

A comprehensive Deep Learning pipeline for the semantic segmentation and morphological analysis of blood vessels from angiography images. This repository implements two state-of-the-art segmentation architectures—**UNet++** and **DeepLabV3+**—using Keras/TensorFlow.

Additionally, it provides tools for **vessel counting and feature extraction** via skeletonization.

---

## 🚀 Features

- **Advanced Architectures**: Full implementations of [UNet++](https://arxiv.org/abs/1807.10165) (with dense skip pathways and optional ResNet50 backbone) and [DeepLabV3+](https://arxiv.org/abs/1802.02611) (with Atrous Spatial Pyramid Pooling).
- **Custom Loss Functions**: Implementations of combined loss functions such as `BCE + Dice` and `Focal + Dice` for handling severe class imbalance in vessel pixels.
- **Robust Metrics**: Built-in tracking of Intersection over Union (IoU), Dice Coefficient, Precision, and Recall.
- **Vessel Analysis**: Post-processing scripts to skeletonize predicted vessel masks, count distinct vessel segments, and calculate vessel lengths.
- **Data Augmentation**: Integrated with `Albumentations` for dynamic, heavy data augmentation during training.

---

## 📂 Project Structure

```text
angio/
│
├── deeplabv3/               # DeepLabV3+ Implementation
│   ├── dataset.py           # Data loading & augmentation
│   ├── model.py             # DeepLabV3+ architecture definition
│   ├── train.py             # Training loop
│   ├── predict.py           # Basic inference script
│   ├── predict_overlay.py   # Inference with GT/Prediction visual overlays
│   └── vessel_analysis.py   # Downstream vessel analysis tools
│
├── unet++/                  # UNet++ Implementation
│   ├── src/                 # Source code (unetpp.py, losses, metrics)
│   ├── train.py             # Training loop
│   ├── predict.py           # Inference & visualisation
│   └── skeletonize_and_count.py # Skeletonization for vessel counting
│
├── 500_rgb_mask/            # Default Training Dataset directory (expected)
├── 137_rgb_mask/            # Default Testing Dataset directory (expected)
└── README.md                # Project documentation
```

---

## 📊 Dataset Requirements

The code expects images and masks structured in specific directories. The standard directories are `500_rgb_mask` (Train) and `137_rgb_mask` (Test). Inside these, images should be under an `RGB/` subfolder, and masks should follow a specific naming convention (e.g., `gauss_NNN_RGB.jpg` for masks corresponding to `NNN_RGB.jpg`).

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/codeninjartr/angio.git
   cd angio
   ```

2. Create a virtual environment (Optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies (Refer to individual folder requirements):
   ```bash
   cd unet++
   pip install -r requirements.txt
   ```

---

## 🧠 Training Models

### Train UNet++
```bash
cd unet++
python src/train.py
```
*Note: You can toggle between a Scratch UNet++ and a ResNet50-backed UNet++ by editing `USE_PRETRAINED` inside `src/train.py`.*

### Train DeepLabV3+
```bash
cd deeplabv3
python train.py
```

Checkpoints will be saved automatically to `models/best_model.keras` based on validation loss.

---

## 🔍 Inference & Visualization

To generate predictions on the test set:

**UNet++:**
```bash
cd unet++
python predict.py
```

**DeepLabV3+:**
```bash
cd deeplabv3
python predict_overlay.py
```

Outputs, including prediction masks, visual overlays, and evaluation metrics, will be saved to the respective `outputs/` directory.

---

## 🔬 Vessel Counting & Skeletonization

Once masks are predicted, you can extract morphological vessel data:

```bash
cd unet++
python skeletonize_and_count.py
```

This will perform skeletonization on the predicted binary masks, locate branch points/endpoints, count distinct vessel segments, and output a CSV report (e.g., `outputs/vessel_counts.csv`).

---

## 📜 License
*Please specify the license for this repository.*
