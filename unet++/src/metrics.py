"""
metrics.py – Evaluation Metrics for UNet++ Vessel Segmentation
===============================================================
Provides:
  • dice_coeff  – Soft Dice coefficient (smooth, used during training)
  • iou_score   – Hard IoU metric (thresholded at 0.5)
  • precision   – Precision metric
  • recall      – Recall metric
"""

import tensorflow as tf


# ─────────────────────────────────────────────
# Dice Coefficient (Soft – for training monitoring)
# ─────────────────────────────────────────────
def dice_coeff(y_true, y_pred):
    """
    Soft Dice coefficient – used as a monitoring metric during training.
    Does NOT threshold predictions; directly uses probabilities.

    Args:
        y_true : ground-truth mask  (batch, H, W, 1),  float32 in {0, 1}
        y_pred : sigmoid prediction (batch, H, W, 1),  float32 in [0, 1]
    Returns:
        scalar Dice in [0, 1]
    """
    smooth = 1e-6

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


# ─────────────────────────────────────────────
# IoU (Intersection over Union) – Hard Metric
# ─────────────────────────────────────────────
def iou_score(y_true, y_pred):
    """
    Hard IoU metric – thresholds predictions at 0.5 before computing.

    IoU = (Intersection + ε) / (Union + ε)

    Args:
        y_true : ground-truth mask  (batch, H, W, 1),  float32 in {0, 1}
        y_pred : sigmoid prediction (batch, H, W, 1),  float32 in [0, 1]
    Returns:
        scalar IoU in [0, 1]
    """
    smooth = 1e-6

    y_pred_bin   = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union        = (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred_bin)
        - intersection
    )

    return (intersection + smooth) / (union + smooth)


# ─────────────────────────────────────────────
# Precision
# ─────────────────────────────────────────────
def precision(y_true, y_pred):
    """
    Hard precision metric – thresholds predictions at 0.5.

    Precision = TP / (TP + FP + ε)

    Args:
        y_true : ground-truth mask  (batch, H, W, 1)
        y_pred : sigmoid prediction (batch, H, W, 1)
    Returns:
        scalar precision in [0, 1]
    """
    smooth     = 1e-6
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred_bin)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred_bin)

    return (tp + smooth) / (tp + fp + smooth)


# ─────────────────────────────────────────────
# Recall
# ─────────────────────────────────────────────
def recall(y_true, y_pred):
    """
    Hard recall metric – thresholds predictions at 0.5.

    Recall = TP / (TP + FN + ε)

    Args:
        y_true : ground-truth mask  (batch, H, W, 1)
        y_pred : sigmoid prediction (batch, H, W, 1)
    Returns:
        scalar recall in [0, 1]
    """
    smooth     = 1e-6
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred_bin)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred_bin))

    return (tp + smooth) / (tp + fn + smooth)
