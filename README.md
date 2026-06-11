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

## 📊 Quantitative Evaluation Results

The models were evaluated on the independent hold-out test set of 137 images. Due to the severe class imbalance in vessel segmentation (vessel pixels comprise only ~5-10% of the image), **Intersection over Union (IoU)** and **Dice Coefficient (F1-Score)** are the primary metrics for validating segmentation quality. Global **Pixel Accuracy** is also tracked for completeness, though it is recognized as highly misleading under extreme class imbalance.

| Architecture | Loss Function | Pixel Accuracy | Mean IoU | Mean Dice |
| :--- | :---: | :---: | :---: | :---: |
| **UNet (Baseline)** | BCE | 0.9412 | 0.7645 | 0.8665 |
| **UNet++ (Scratch)** | BCE + Dice | 0.9588 | 0.8210 | 0.9017 |
| **UNet++ (ResNet50)** | BCE + Dice | **0.9631** | **0.8605** | **0.9250** |
| **DeepLabV3+** | BCE + Dice | 0.9590 | 0.8547 | 0.9216 |

---

## 🔬 Vessel Counting & Skeletonization

Once masks are predicted, you can extract morphological vessel data:

```bash
cd unet++
python skeletonize_and_count.py
```

This will perform skeletonization on the predicted binary masks, locate branch points/endpoints, count distinct vessel segments, and output a CSV report (e.g., `outputs/vessel_counts.csv`).

---

## 📈 Vessel Growth Analysis (Multi-Concentration Comparison)

To analyze how egg embryo/angiogenesis blood vessels grow and branch over time across different drug concentrations, we performed a quantitative growth analysis for **Control**, **0.1ug**, **1ug**, and **10ug** concentrations. 

### Methodology
1. **Metadata Decoding**: Dataset images (500 train, 137 test) were matched back to the original cropped files in the raw dataset using downscaled MSE similarity to extract experimental metadata (concentration and timepoints).
2. **Segmentation**: Predicted high-fidelity vessel masks for the matched images using our best-performing **UNet++ (ResNet50)** model.
3. **Skeletonization & Feature Extraction**: Skeletonized the segmentation masks and ran neighborhood convolution to extract:
   - **Vessel Networks**: Count of distinct connected vascular trees.
   - **Branch Points**: Locations where vessels branch/bifurcate (neighbor count > 2).
   - **End Points**: Vessel endpoints/terminations (neighbor count = 1).
4. **Aggregation**: Aggregated metrics by concentration and time point (0h, 2h, 4h, 8h, 24h, 32h) to compute mean and standard deviation.

### Summary of Results

| Concentration | Time Point | Sample Size (n) | Vessel Networks (Mean ± SD) | Branch Points (Mean ± SD) | End Points (Mean ± SD) |
|---|---|---|---|---|---|
| **CONTROL** | 0 hours | 15 | 11.47 ± 3.52 | 2279.53 ± 814.91 | 52.27 ± 15.64 |
| | 2 hours | 11 | 14.82 ± 2.72 | 2917.27 ± 761.50 | 54.00 ± 8.82 |
| | 4 hours | 6 | 10.67 ± 3.40 | 2538.00 ± 578.08 | 45.50 ± 11.79 |
| | 8 hours | 5 | 14.00 ± 2.00 | 2724.00 ± 476.45 | 63.20 ± 16.17 |
| | 24 hours | 9 | 13.00 ± 3.33 | 2864.22 ± 791.19 | 48.11 ± 5.04 |
| **0.1ug** | 0 hours | 23 | 12.61 ± 4.52 | 2667.83 ± 781.62 | 46.43 ± 9.46 |
| | 2 hours | 30 | 11.73 ± 3.82 | 3033.93 ± 1065.80 | 47.37 ± 10.44 |
| | 4 hours | 31 | 11.74 ± 4.60 | 2725.23 ± 923.63 | 48.58 ± 11.42 |
| | 8 hours | 18 | 10.78 ± 3.58 | 2882.44 ± 956.03 | 48.94 ± 12.49 |
| | 24 hours | 22 | 12.50 ± 5.08 | 2948.00 ± 866.08 | 48.55 ± 9.97 |
| | 32 hours | 2 | 13.00 ± 2.00 | 2375.00 ± 209.00 | 58.50 ± 2.50 |
| **1ug** | 0 hours | 77 | 11.57 ± 3.96 | 2938.65 ± 896.46 | 50.34 ± 13.24 |
| | 2 hours | 46 | 11.65 ± 3.73 | 2605.83 ± 921.93 | 48.09 ± 10.21 |
| | 4 hours | 45 | 12.67 ± 3.13 | 2719.98 ± 846.45 | 48.53 ± 10.88 |
| | 8 hours | 52 | 11.94 ± 3.21 | 2760.40 ± 806.49 | 47.50 ± 10.43 |
| | 24 hours | 25 | 13.60 ± 4.34 | 2529.24 ± 820.60 | 51.40 ± 7.38 |
| | 32 hours | 2 | 7.50 ± 1.50 | 2531.50 ± 230.50 | 34.00 ± 5.00 |
| **10ug** | 0 hours | 49 | 11.49 ± 4.20 | 3032.92 ± 1211.44 | 50.02 ± 13.81 |
| | 2 hours | 50 | 12.34 ± 4.34 | 2625.70 ± 1097.52 | 49.40 ± 11.52 |
| | 4 hours | 42 | 12.50 ± 4.38 | 2675.95 ± 756.63 | 49.62 ± 11.40 |
| | 8 hours | 49 | 11.73 ± 4.23 | 2674.55 ± 888.43 | 47.29 ± 10.29 |
| | 24 hours | 28 | 13.29 ± 3.10 | 2789.75 ± 733.40 | 49.79 ± 9.84 |

### Visualizations

The script `vessel_growth_analysis.py` generates the following assets:
- **`outputs/vessel_growth_plots.png`**: Comparative progression curves for networks, branches, and endpoints over time.
- **`outputs/vessel_growth_table.png`**: Styled publication-ready results table.

You can run the analysis using:
```bash
python vessel_growth_analysis.py
```

---

## 📜 License
*Please specify the license for this repository.*
