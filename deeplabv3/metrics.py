import tensorflow as tf

def iou_score(y_true, y_pred):
    """
    Computes the Intersection over Union (IoU) / Jaccard Index for binary segmentation.
    IoU = Area of Overlap / Area of Union
    """
    smooth = 1e-6

    # Apply thresholding to prediction
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) -
        intersection
    )

    return (intersection + smooth) / (union + smooth)
