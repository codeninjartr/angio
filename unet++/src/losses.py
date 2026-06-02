"""
losses.py – Loss Functions for UNet++ Vessel Segmentation
===========================================================
Provides:
  • dice_loss       – Soft Dice loss (handles class imbalance well)
  • bce_dice_loss   – BCE + Dice (recommended for vessel segmentation)
  • focal_dice_loss – Focal + Dice (for highly imbalanced vessel pixels)
"""

import tensorflow as tf


# ─────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────
def dice_loss(y_true, y_pred):
    """
    Soft Dice loss – works well for class-imbalanced vessel segmentation.

    Args:
        y_true : ground-truth mask  (batch, H, W, 1)
        y_pred : sigmoid prediction (batch, H, W, 1)
    Returns:
        scalar loss in [0, 1]
    """
    smooth = 1e-4   # raised from 1e-6 – prevents loss explosions when
                    # predictions briefly collapse to ~0 during val

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return 1.0 - (
        (2.0 * intersection + smooth) /
        (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    )


# ─────────────────────────────────────────────
# BCE + Dice (Combined) – Primary Loss
# ─────────────────────────────────────────────
def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Cross-Entropy + Dice loss.

    BCE handles per-pixel probability calibration;
    Dice handles overlap / class imbalance.

    Args:
        y_true : ground-truth mask  (batch, H, W, 1)
        y_pred : sigmoid prediction (batch, H, W, 1)
    Returns:
        scalar combined loss
    """
    bce    = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)

    return bce + d_loss


# ─────────────────────────────────────────────
# Focal + Dice – Alternative for Extreme Imbalance
# ─────────────────────────────────────────────
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss – down-weights easy negatives to focus on hard vessel pixels.

    Args:
        y_true : ground-truth mask  (batch, H, W, 1)
        y_pred : sigmoid prediction (batch, H, W, 1)
        gamma  : focusing parameter (default 2.0)
        alpha  : class balancing weight (default 0.25)
    Returns:
        scalar focal loss
    """
    y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    bce     = -( y_true * tf.math.log(y_pred)
               + (1.0 - y_true) * tf.math.log(1.0 - y_pred) )
    p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
    fl      = alpha_t * tf.pow(1.0 - p_t, gamma) * bce

    return tf.reduce_mean(fl)


def focal_dice_loss(y_true, y_pred):
    """
    Combined Focal + Dice loss.

    Args:
        y_true : ground-truth mask  (batch, H, W, 1)
        y_pred : sigmoid prediction (batch, H, W, 1)
    Returns:
        scalar combined loss
    """
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
