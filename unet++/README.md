# UNet++ – Vessel Segmentation

A pure-Keras **UNet++** implementation for binary vessel segmentation, mirroring the project layout and coding conventions of the adjacent `deeplabv3+/` project.

---

## Project Structure

```
unet++/
│
├── src/
│   ├── dataset.py     – Data loading, Albumentations augmentation
│   ├── losses.py      – Dice, BCE+Dice, Focal+Dice losses
│   ├── metrics.py     – Dice coeff, IoU, Precision, Recall
│   ├── unetpp.py      – UNet++ architecture (scratch + ResNet50)
│   └── train.py       – Full training pipeline
│
├── models/            – Saved checkpoints (best_model.keras)
├── outputs/           – Plots, CSV logs, predicted masks
├── predict.py         – Inference & visualization script
└── requirements.txt
```

---

## Dataset

Same paths as `deeplabv3+`:

| Split | Folder | Images | Masks |
|-------|--------|--------|-------|
| Train | `500_rgb_mask/RGB/` | `NNN_RGB.jpg` | `gauss_NNN_RGB.jpg` |
| Test  | `137_rgb_mask/RGB/` | `NNN_RGB.jpg` | `gauss_NNN_RGB.jpg` |

---

## Install

```bash
pip install -r requirements.txt
```

---

## Train

```bash
cd d:\labdatanew@aniket\unet++
python src/train.py
```

Switch between model variants in `src/train.py`:

```python
USE_PRETRAINED = True   # ResNet50 encoder (recommended for ~500 images)
USE_PRETRAINED = False  # Scratch UNet++ (lighter, faster)
```

---

## Inference

```bash
python predict.py
```

Outputs saved to `outputs/`:
- `predictions.png` – 4-panel grid (RGB / GT / Prob map / Overlay)
- `pred_masks/`     – per-sample binary PNG files

---

## UNet++ Architecture

### Dense Skip Pathways (Xij Grid)

```
X00 ── X01 ── X02 ── X03 ── X04   ← output
 │      │      │      │
X10 ── X11 ── X12 ── X13
 │      │      │
X20 ── X21 ── X22
 │      │
X30 ── X31
 │
X40  (bottleneck)
```

Each node **Xij** receives:
1. All same-scale predecessors `Xi(j-1) … Xi0` (dense skip connections)
2. The upsampled output of `X(i+1)(j-1)` (cross-scale)

This is the key difference from vanilla UNet's single skip connection per level.

### Two Variants

| Variant | Encoder | Pretrained | Best for |
|---------|---------|-----------|----------|
| `UNetPlusPlus` | Scratch (5-level) | No | Baseline |
| `UNetPlusPlusResNet50` | ResNet50 | ImageNet | ~500 images (recommended) |

---

## Hyperparameters

| Parameter   | Default | Notes |
|-------------|---------|-------|
| `IMG_SIZE`  | 256     | 512 for RTX/A100 |
| `BATCH_SIZE`| 4       | Reduce if OOM |
| `EPOCHS`    | 100     | EarlyStopping active |
| `LR`        | 1e-4    | ReduceLROnPlateau active |
| `DROPOUT`   | 0.2     | Helps generalise on small datasets |

---

## Outputs

| File | Description |
|------|-------------|
| `models/best_model.keras` | Best checkpoint (val_loss) |
| `outputs/training_log.csv` | Per-epoch metrics |
| `outputs/training_curve.png` | 4-panel training curves |
| `outputs/tb_logs/` | TensorBoard logs |
| `outputs/predictions.png` | Visualisation grid |
| `outputs/pred_masks/` | Binary predicted masks |
