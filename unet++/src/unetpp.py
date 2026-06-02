"""
unetpp.py – UNet++ Architecture (from scratch, pure Keras/TensorFlow)
======================================================================
Architecture
  • Encoder  : ResNet50 (ImageNet pre-trained, stride-4 low-level + stride-16 high-level)
  • Decoder  : Dense nested skip pathways (UNet++ / UNet3+)
  • Output   : 1×1 Conv + Sigmoid → binary probability map

UNet++ Grid Layout (Xij notation):
  X00 ── X01 ── X02 ── X03
   │      │      │
  X10 ── X11 ── X12
   │      │
  X20 ── X21
   │
  X30

Each intermediate node Xij aggregates:
  1. All same-scale predecessors  Xi0 … Xi(j-1)  (dense skip connections)
  2. The down-sampled output of   X(i+1)(j-1)   (up-path)

Reference: Zhou et al., 2018 – "UNet++: A Nested U-Net Architecture for
Medical Image Segmentation"
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, UpSampling2D, Concatenate, Dropout
)
from tensorflow.keras.models import Model


# ─────────────────────────────────────────────
# Shared Building Block
# ─────────────────────────────────────────────
def conv_block(x, filters, dropout_rate=0.0, name_prefix=""):
    """
    Two consecutive Conv → BN → ReLU blocks (standard UNet cell).

    Args:
        x            : input KerasTensor
        filters      : number of output feature maps
        dropout_rate : optional spatial dropout between blocks
        name_prefix  : string prefix for unique layer names
    Returns:
        KerasTensor  (same spatial size, `filters` channels)
    """
    x = Conv2D(
        filters, 3,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv1" if name_prefix else None
    )(x)
    x = BatchNormalization(
        name=f"{name_prefix}_bn1" if name_prefix else None
    )(x)
    x = Activation(
        "relu",
        name=f"{name_prefix}_relu1" if name_prefix else None
    )(x)

    if dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)

    x = Conv2D(
        filters, 3,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv2" if name_prefix else None
    )(x)
    x = BatchNormalization(
        name=f"{name_prefix}_bn2" if name_prefix else None
    )(x)
    x = Activation(
        "relu",
        name=f"{name_prefix}_relu2" if name_prefix else None
    )(x)

    return x


# ─────────────────────────────────────────────
# UNet++ Model (scratch encoder, 4-level)
# ─────────────────────────────────────────────
def UNetPlusPlus(input_shape=(256, 256, 3),
                 filters=(32, 64, 128, 256, 512),
                 dropout_rate=0.2):
    """
    UNet++ with dense nested skip pathways – pure Keras implementation.

    Grid of nodes (i = depth, j = dense step):
        X00  X01  X02  X03  X04
        X10  X11  X12  X13
        X20  X21  X22
        X30  X31
        X40

    Final output is taken from X04 (full-resolution reconstruction).

    Args:
        input_shape  : (H, W, C)  – default (256, 256, 3)
        filters      : channel widths at each encoder level
        dropout_rate : dropout between conv blocks (helps small datasets)
    Returns:
        tf.keras.Model
    """
    inputs = Input(shape=input_shape, name="rgb_input")

    f = filters   # shorthand

    # ── COLUMN 0  (Encoder / down-path) ──────────────────────────────────────
    # X00  (level 0 – full resolution)
    x00 = conv_block(inputs, f[0], dropout_rate, name_prefix="x00")

    # X10  (level 1 – stride 2)
    p0  = MaxPooling2D((2, 2), name="pool0")(x00)
    x10 = conv_block(p0, f[1], dropout_rate, name_prefix="x10")

    # X20  (level 2 – stride 4)
    p1  = MaxPooling2D((2, 2), name="pool1")(x10)
    x20 = conv_block(p1, f[2], dropout_rate, name_prefix="x20")

    # X30  (level 3 – stride 8)
    p2  = MaxPooling2D((2, 2), name="pool2")(x20)
    x30 = conv_block(p2, f[3], dropout_rate, name_prefix="x30")

    # X40  (level 4 – stride 16, bottleneck)
    p3  = MaxPooling2D((2, 2), name="pool3")(x30)
    x40 = conv_block(p3, f[4], dropout_rate, name_prefix="x40")

    # ── COLUMN 1  (First dense step) ─────────────────────────────────────────
    # X01 : up(X10) + X00
    x01 = Concatenate(name="cat_x01")(
        [x00, UpSampling2D((2, 2), interpolation="bilinear", name="up_x10_to_x01")(x10)]
    )
    x01 = conv_block(x01, f[0], dropout_rate, name_prefix="x01")

    # X11 : up(X20) + X10
    x11 = Concatenate(name="cat_x11")(
        [x10, UpSampling2D((2, 2), interpolation="bilinear", name="up_x20_to_x11")(x20)]
    )
    x11 = conv_block(x11, f[1], dropout_rate, name_prefix="x11")

    # X21 : up(X30) + X20
    x21 = Concatenate(name="cat_x21")(
        [x20, UpSampling2D((2, 2), interpolation="bilinear", name="up_x30_to_x21")(x30)]
    )
    x21 = conv_block(x21, f[2], dropout_rate, name_prefix="x21")

    # X31 : up(X40) + X30
    x31 = Concatenate(name="cat_x31")(
        [x30, UpSampling2D((2, 2), interpolation="bilinear", name="up_x40_to_x31")(x40)]
    )
    x31 = conv_block(x31, f[3], dropout_rate, name_prefix="x31")

    # ── COLUMN 2  (Second dense step) ────────────────────────────────────────
    # X02 : up(X11) + X00 + X01   (dense: all same-level predecessors)
    x02 = Concatenate(name="cat_x02")(
        [x00, x01, UpSampling2D((2, 2), interpolation="bilinear", name="up_x11_to_x02")(x11)]
    )
    x02 = conv_block(x02, f[0], dropout_rate, name_prefix="x02")

    # X12 : up(X21) + X10 + X11
    x12 = Concatenate(name="cat_x12")(
        [x10, x11, UpSampling2D((2, 2), interpolation="bilinear", name="up_x21_to_x12")(x21)]
    )
    x12 = conv_block(x12, f[1], dropout_rate, name_prefix="x12")

    # X22 : up(X31) + X20 + X21
    x22 = Concatenate(name="cat_x22")(
        [x20, x21, UpSampling2D((2, 2), interpolation="bilinear", name="up_x31_to_x22")(x31)]
    )
    x22 = conv_block(x22, f[2], dropout_rate, name_prefix="x22")

    # ── COLUMN 3  (Third dense step) ─────────────────────────────────────────
    # X03 : up(X12) + X00 + X01 + X02
    x03 = Concatenate(name="cat_x03")(
        [x00, x01, x02, UpSampling2D((2, 2), interpolation="bilinear", name="up_x12_to_x03")(x12)]
    )
    x03 = conv_block(x03, f[0], dropout_rate, name_prefix="x03")

    # X13 : up(X22) + X10 + X11 + X12
    x13 = Concatenate(name="cat_x13")(
        [x10, x11, x12, UpSampling2D((2, 2), interpolation="bilinear", name="up_x22_to_x13")(x22)]
    )
    x13 = conv_block(x13, f[1], dropout_rate, name_prefix="x13")

    # ── COLUMN 4  (Full reconstruction) ──────────────────────────────────────
    # X04 : up(X13) + X00 + X01 + X02 + X03
    x04 = Concatenate(name="cat_x04")(
        [x00, x01, x02, x03, UpSampling2D((2, 2), interpolation="bilinear", name="up_x13_to_x04")(x13)]
    )
    x04 = conv_block(x04, f[0], dropout_rate, name_prefix="x04")

    # ── Output head ───────────────────────────────────────────────────────────
    outputs = Conv2D(1, 1, activation="sigmoid", name="seg_output")(x04)

    model = Model(inputs=inputs, outputs=outputs, name="UNetPlusPlus")

    return model


# ─────────────────────────────────────────────
# UNet++ with ResNet50 Encoder (Pretrained)
# ─────────────────────────────────────────────
def UNetPlusPlusResNet50(input_shape=(256, 256, 3), dropout_rate=0.2):
    """
    UNet++ decoder grafted onto a frozen ImageNet-pretrained ResNet50 encoder.

    Encoder feature maps (for 256×256 input):
        Layer                   Stride   Spatial size   Channels
        ──────────────────────  ──────   ────────────   ────────
        conv1_relu              /2       128×128        64
        conv2_block3_out        /4       64×64          256
        conv3_block4_out        /8       32×32          512
        conv4_block6_out        /16      16×16          1024
        conv5_block3_out        /32      8×8            2048

    Dense decoder mirrors the 4-level UNet++ grid using upsampled
    encoder features as the "column-0" nodes.

    Args:
        input_shape  : (H, W, C) – default (256, 256, 3)
        dropout_rate : dropout between conv blocks
    Returns:
        tf.keras.Model
    """
    inputs = Input(shape=input_shape, name="rgb_input")

    # ── ResNet50 backbone ─────────────────────────────────────────────────────
    backbone = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    # Freeze early layers (good for small datasets ≤ 500 images)
    for layer in backbone.layers[:80]:
        layer.trainable = False

    # ── Extract multi-scale encoder features ──────────────────────────────────
    # s2  : 128×128   (stride /2)
    s2  = backbone.get_layer("conv1_relu").output          # 64 ch
    # s4  : 64×64     (stride /4)
    s4  = backbone.get_layer("conv2_block3_out").output    # 256 ch
    # s8  : 32×32     (stride /8)
    s8  = backbone.get_layer("conv3_block4_out").output    # 512 ch
    # s16 : 16×16     (stride /16)
    s16 = backbone.get_layer("conv4_block6_out").output    # 1024 ch
    # s32 : 8×8       (stride /32)  – bottleneck
    s32 = backbone.get_layer("conv5_block3_out").output    # 2048 ch

    DEC = 128   # uniform decoder channel width

    # Project all encoder features to DEC channels
    e0 = conv_block(s2,  DEC, dropout_rate, "enc0")   # 128×128
    e1 = conv_block(s4,  DEC, dropout_rate, "enc1")   # 64×64
    e2 = conv_block(s8,  DEC, dropout_rate, "enc2")   # 32×32
    e3 = conv_block(s16, DEC, dropout_rate, "enc3")   # 16×16
    e4 = conv_block(s32, DEC, dropout_rate, "enc4")   # 8×8

    # ── UNet++ Dense decoder ──────────────────────────────────────────────────
    def up(x):
        return UpSampling2D((2, 2), interpolation="bilinear")(x)

    # ─── Column 1 ───
    d01 = conv_block(Concatenate()([e0, up(e1)]), DEC, dropout_rate, "d01")
    d11 = conv_block(Concatenate()([e1, up(e2)]), DEC, dropout_rate, "d11")
    d21 = conv_block(Concatenate()([e2, up(e3)]), DEC, dropout_rate, "d21")
    d31 = conv_block(Concatenate()([e3, up(e4)]), DEC, dropout_rate, "d31")

    # ─── Column 2 ───
    d02 = conv_block(Concatenate()([e0, d01, up(d11)]), DEC, dropout_rate, "d02")
    d12 = conv_block(Concatenate()([e1, d11, up(d21)]), DEC, dropout_rate, "d12")
    d22 = conv_block(Concatenate()([e2, d21, up(d31)]), DEC, dropout_rate, "d22")

    # ─── Column 3 ───
    d03 = conv_block(Concatenate()([e0, d01, d02, up(d12)]), DEC, dropout_rate, "d03")
    d13 = conv_block(Concatenate()([e1, d11, d12, up(d22)]), DEC, dropout_rate, "d13")

    # ─── Column 4 (final) ───
    d04 = conv_block(Concatenate()([e0, d01, d02, d03, up(d13)]), DEC, dropout_rate, "d04")

    # ── Final upsample × 2 → original 256×256 resolution ────────────────────
    # e0 is at 128×128; one more ×2 brings it to 256×256
    d04_full = up(d04)

    # ── Output head ───────────────────────────────────────────────────────────
    outputs = Conv2D(1, 1, activation="sigmoid", name="seg_output")(d04_full)

    model = Model(inputs=inputs, outputs=outputs, name="UNetPlusPlus_ResNet50")

    return model


# ─────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    print("\n─── Scratch UNet++ ───────────────────────")
    m1 = UNetPlusPlus(input_shape=(256, 256, 3))
    m1.summary(line_length=100)
    dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
    out1  = m1.predict(dummy, verbose=0)
    print(f"Input : {dummy.shape}  →  Output: {out1.shape}")

    print("\n─── ResNet50 UNet++ ──────────────────────")
    m2 = UNetPlusPlusResNet50(input_shape=(256, 256, 3))
    m2.summary(line_length=100)
    out2  = m2.predict(dummy, verbose=0)
    print(f"Input : {dummy.shape}  →  Output: {out2.shape}")
