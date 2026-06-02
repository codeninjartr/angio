import tensorflow as tf

def dice_loss(y_true, y_pred):
    """
    Computes the Dice Loss for binary segmentation.
    Dice = 2 * (A ∩ B) / (A + B)
    """
    smooth = 1e-6

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return 1.0 - (
        (2.0 * intersection + smooth) /
        (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    )

def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Crossentropy and Dice Loss.
    Recommended for vessel segmentation tasks.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss
