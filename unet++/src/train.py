"""
train.py – UNet++ Training Script
===================================
Usage:
    python train.py

Outputs:
    models/best_model.keras           – best checkpoint (by val_loss)
    outputs/training_log.csv          – per-epoch metrics (Keras CSVLogger)
    outputs/training_curve.png        – loss & metric plots
    outputs/tb_logs/                  – TensorBoard logs
    outputs/terminal_YYYYMMDD_HHMMSS.log  – full terminal output (auto-saved)

Architecture choices (edit below):
    USE_PRETRAINED = True   → UNet++ with ResNet50 encoder (recommended)
    USE_PRETRAINED = False  → UNet++ scratch encoder (faster, lighter)
"""

import os
import sys
import datetime
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – works on servers
import matplotlib.pyplot as plt

# ── Resolve paths relative to this file so the script works from any cwd ─────
# train.py lives at:  unet++/src/train.py
# ROOT_DIR  points to: unet++/
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))   # …/unet++/src
ROOT_DIR = os.path.dirname(SRC_DIR)                     # …/unet++

# Add src/ to path so sibling modules (dataset, unetpp, …) are importable
sys.path.insert(0, SRC_DIR)

from dataset import load_train_data, load_test_data
from unetpp  import UNetPlusPlus, UNetPlusPlusResNet50
from losses  import bce_dice_loss
from metrics import dice_coeff, iou_score, precision, recall

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# Hyper-parameters  ← edit these
# ─────────────────────────────────────────────
BATCH_SIZE      = 4
EPOCHS          = 100
LR              = 5e-5      # ↓ from 1e-4  (val collapses showed 1e-4 is too aggressive)
INPUT_SHAPE     = (256, 256, 3)
DROPOUT_RATE    = 0.35     # ↑ from 0.2  (train dice 0.93 vs val 0.02 = severe overfit)

# True  → ResNet50 encoder (ImageNet weights, recommended for ~500 images)
# False → Scratch UNet++ encoder (no pretrained weights)
USE_PRETRAINED  = True

# Freeze the first N backbone layers – higher = less risk of catastrophic forgetting
# Previous run used 80; raising to 120 keeps more ResNet blocks frozen on 500 images
FREEZE_LAYERS   = 120

# ─────────────────────────────────────────────
# TeeLogger – mirrors terminal output to a file
# ─────────────────────────────────────────────
class TeeLogger:
    """
    Redirects sys.stdout (and optionally sys.stderr) so that every line
    printed to the terminal is ALSO written to a log file.

    Usage:
        logger = TeeLogger("outputs/terminal_20240101_120000.log")
        ...  # all print() calls are captured
        logger.close()   # restore original stdout/stderr
    """

    def __init__(self, filepath: str, mode: str = "w"):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.logfile  = open(filepath, mode, encoding="utf-8", buffering=1)

        # Reconfigure the real terminal to UTF-8 so Unicode chars (arrows,
        # checkmarks) don't crash on Windows cp1252 consoles (Python >= 3.7)
        try:
            self.terminal.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass   # older Python / non-TextIOWrapper -- handled in write()

        sys.stdout = self          # redirect stdout
        sys.stderr = self          # redirect stderr (Keras progress bars)
        print(f"[TeeLogger] Saving terminal output to:\n            {filepath}")

    def write(self, message):
        # Always write full UTF-8 to the log file
        self.logfile.write(message)
        # Write to terminal -- replace unencodable chars instead of crashing
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            enc  = getattr(self.terminal, "encoding", None) or "utf-8"
            safe = message.encode(enc, errors="replace").decode(enc)
            self.terminal.write(safe)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        self.logfile.close()
        print(f"[TeeLogger] Log file closed.")

    # Make it usable as a context manager too
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# Paths  (all anchored to ROOT_DIR = unet++/)
# ─────────────────────────────────────────────
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Start capturing terminal output
_run_tag    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path   = os.path.join(OUTPUTS_DIR, f"terminal_{_run_tag}.log")
_tee_logger = TeeLogger(_log_path)

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("\n[1/5] Loading dataset …")
print("      TRAIN ← d:/labdatanew@aniket/500_rgb_mask/")
print("      TEST  ← d:/labdatanew@aniket/137_rgb_mask/")

X_train, Y_train = load_train_data()
X_test,  Y_test  = load_test_data()

print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 2. tf.data Augmentation Pipeline
# ─────────────────────────────────────────────
def augment(image, mask):
    """
    Online augmentation via tf.data (fast, GPU-compatible).
    Albumentations augmentation was applied at load time (optional).
    """
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)

    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)

    # Random brightness / contrast (image only)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def build_dataset(X, Y, augment_data=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = build_dataset(X_train, Y_train, augment_data=True,  shuffle=True)
val_ds   = build_dataset(X_test,  Y_test,  augment_data=False, shuffle=False)

# ─────────────────────────────────────────────
# 3. Build Model
# ─────────────────────────────────────────────
print("\n[2/5] Building UNet++ model …")

if USE_PRETRAINED:
    print(f"      Variant       : UNet++ with ResNet50 encoder (ImageNet)")
    print(f"      Freeze layers : first {FREEZE_LAYERS} backbone layers")
    model = UNetPlusPlusResNet50(
        input_shape=INPUT_SHAPE,
        dropout_rate=DROPOUT_RATE
    )
    # Apply FREEZE_LAYERS setting (overrides the default 80 set inside the function)
    for layer in model.layers:
        layer.trainable = True          # unfreeze all first
    frozen = 0
    for layer in model.layers:
        if hasattr(layer, 'layers'):    # it's the ResNet50 sub-model
            for sub in layer.layers[:FREEZE_LAYERS]:
                sub.trainable = False
                frozen += 1
    print(f"      Frozen params : {frozen} backbone sub-layers")
else:
    print("      Variant : UNet++ scratch encoder")
    model = UNetPlusPlus(
        input_shape=INPUT_SHAPE,
        dropout_rate=DROPOUT_RATE
    )

model.summary(line_length=110)
trainable     = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
print(f"\n      Trainable parameters     : {trainable:,}")
print(f"      Non-trainable parameters : {non_trainable:,}")

# ─────────────────────────────────────────────
# 4. Compile
# ─────────────────────────────────────────────
print("\n[3/5] Compiling ...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=bce_dice_loss,
    metrics=["accuracy", dice_coeff, iou_score, precision, recall]
)

# ─────────────────────────────────────────────
# Resume Detection
# Checks for an existing checkpoint and the CSV
# log to continue from where training stopped.
# ─────────────────────────────────────────────
BEST_CKPT = os.path.join(MODELS_DIR,  "best_model.keras")
LAST_CKPT = os.path.join(MODELS_DIR,  "last_checkpoint.keras")
CSV_LOG   = os.path.join(OUTPUTS_DIR, "training_log.csv")

INITIAL_EPOCH = 0
RESUMING      = False

# Pick the most recent checkpoint to load weights from
_ckpt_to_load = None
if os.path.exists(LAST_CKPT):
    _ckpt_to_load = LAST_CKPT          # prefer the most recent epoch
elif os.path.exists(BEST_CKPT):
    _ckpt_to_load = BEST_CKPT          # fall back to best saved

if _ckpt_to_load and os.path.exists(CSV_LOG):
    import pandas as pd
    try:
        _df = pd.read_csv(CSV_LOG)
        if len(_df) > 0:
            INITIAL_EPOCH = int(_df["epoch"].max()) + 1
            model.load_weights(_ckpt_to_load)
            RESUMING = True
            print(f"\n[RESUME] Loaded weights  : {_ckpt_to_load}")
            print(f"[RESUME] Resuming from   : Epoch {INITIAL_EPOCH + 1}/{EPOCHS}")
        else:
            print("\n[INFO] CSV log is empty  - starting fresh.")
    except Exception as e:
        print(f"\n[WARNING] Could not read CSV log ({e}) - starting fresh.")
else:
    print("\n[INFO] No checkpoint found - starting fresh.")

# ─────────────────────────────────────────────
# 5. Callbacks
# ─────────────────────────────────────────────
callbacks = [

    # Monitor val_dice_coeff (mode=max) - robust to val_loss spikes caused
    # by the small 137-image validation set
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_CKPT,
        monitor="val_dice_coeff",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Save the latest epoch every 5 epochs so resume always has a recent copy
    tf.keras.callbacks.ModelCheckpoint(
        filepath=LAST_CKPT,
        save_freq="epoch",          # save every epoch
        save_best_only=False,
        verbose=0
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coeff",
        mode="max",
        factor=0.5,
        patience=4,
        min_delta=1e-3,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coeff",
        mode="max",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),

    # append=True when resuming so old epoch rows are NOT overwritten
    tf.keras.callbacks.CSVLogger(
        CSV_LOG,
        append=RESUMING
    ),

    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(OUTPUTS_DIR, "tb_logs"),
        histogram_freq=0
    ),
]

# ─────────────────────────────────────────────
# 6. Train
# ─────────────────────────────────────────────
print("\n[4/5] Training ...")
print(f"      Epochs       : {EPOCHS}")
print(f"      Start epoch  : {INITIAL_EPOCH + 1}")
print(f"      Batch size   : {BATCH_SIZE}")
print(f"      LR           : {LR}")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=INITIAL_EPOCH,   # <-- resumes from here, not 0
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────
print("\n[5/5] Evaluating on test set (137_rgb_mask) …")
results      = model.evaluate(val_ds, verbose=1)
metric_names = model.metrics_names

print("\n── Test Results ──────────────────────────")
for name, val in zip(metric_names, results):
    print(f"  {name:20s}: {val:.4f}")

# ─────────────────────────────────────────────
# 8. Plot Training Curves
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("UNet++ Training Curves", fontsize=14, fontweight="bold")

    metrics_to_plot = [
        ("loss",       "Loss (BCE + Dice)"),
        ("dice_coeff", "Dice Coefficient"),
        ("iou_score",  "IoU Score"),
        ("precision",  "Precision"),
    ]

    for ax, (key, title) in zip(axes, metrics_to_plot):
        val_key = "val_" + key
        if key in history.history:
            ax.plot(history.history[key],     label="Train", linewidth=2)
        if val_key in history.history:
            ax.plot(history.history[val_key], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, "training_curve.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n[INFO] Training curve saved → {save_path}")
    plt.close()


plot_history(history)
print(f"\n✅ Training complete. Best model saved → {os.path.join(MODELS_DIR, 'best_model.keras')}")

# ─────────────────────────────────────────────
# Close TeeLogger – flush & restore stdout
# ─────────────────────────────────────────────
_tee_logger.close()
print(f"\n📄 Full terminal log saved → {_log_path}")
