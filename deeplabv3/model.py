import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Concatenate, 
    UpSampling2D, AveragePooling2D
)
from tensorflow.keras.models import Model

def ASPP(inputs):
    """
    Atrous Spatial Pyramid Pooling block.
    Extracts multi-scale contextual information using different dilation rates.
    """
    # 1. 1x1 convolution branch
    y1 = Conv2D(256, 1, padding="same", use_bias=False)(inputs)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)

    # 2. 3x3 convolution with dilation rate = 6
    y2 = Conv2D(
        256,
        3,
        dilation_rate=6,
        padding="same",
        use_bias=False
    )(inputs)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)

    # 3. 3x3 convolution with dilation rate = 12
    y3 = Conv2D(
        256,
        3,
        dilation_rate=12,
        padding="same",
        use_bias=False
    )(inputs)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)

    # 4. 3x3 convolution with dilation rate = 18
    y4 = Conv2D(
        256,
        3,
        dilation_rate=18,
        padding="same",
        use_bias=False
    )(inputs)
    y4 = BatchNormalization()(y4)
    y4 = ReLU()(y4)

    # 5. Image Pooling branch
    # Dynamically pool the entire spatial dimensions of the input
    img_pool = AveragePooling2D(pool_size=(inputs.shape[1], inputs.shape[2]))(inputs)
    img_pool = Conv2D(256, 1, padding="same", use_bias=False)(img_pool)
    img_pool = BatchNormalization()(img_pool)
    img_pool = ReLU()(img_pool)
    
    # Upsample the image pooling branch back to match ASPP branch shapes
    img_pool = UpSampling2D(
        size=(inputs.shape[1], inputs.shape[2]),
        interpolation="bilinear"
    )(img_pool)

    # Concatenate all ASPP branches and the image pooling branch
    y = Concatenate()([y1, y2, y3, y4, img_pool])

    # 1x1 convolution projection and regularization
    y = Conv2D(256, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    return y

def DeepLabV3Plus(input_shape=(256, 256, 3)):
    """
    DeepLabV3+ architecture with a pre-trained ResNet50 backbone.
    """
    inputs = Input(shape=input_shape)

    # Encoder backbone using ResNet50 pre-trained on ImageNet
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    # Low-level features from block 2 (output shape 64x64 for 256x256 input)
    low_level = base_model.get_layer("conv2_block3_out").output

    # High-level features from block 4 (output shape 16x16 for 256x256 input)
    encoder_output = base_model.get_layer("conv4_block6_out").output

    # Atrous Spatial Pyramid Pooling (ASPP)
    x = ASPP(encoder_output)

    # Upsample high-level ASPP features by 4 (to 64x64)
    x = UpSampling2D(
        size=(4, 4),
        interpolation="bilinear"
    )(x)

    # Process low-level features with 1x1 convolution (reduce channel dimension)
    low_level = Conv2D(
        48,
        1,
        padding="same",
        use_bias=False
    )(low_level)
    low_level = BatchNormalization()(low_level)
    low_level = ReLU()(low_level)

    # Concatenate upsampled high-level features and processed low-level features
    x = Concatenate()([x, low_level])

    # Decoder processing (3x3 convolutions with ReLU activation)
    x = Conv2D(
        256,
        3,
        padding="same",
        activation="relu"
    )(x)
    x = Conv2D(
        256,
        3,
        padding="same",
        activation="relu"
    )(x)

    # Final upsampling by 4 to restore resolution to input dimensions (to 256x256)
    x = UpSampling2D(
        size=(4, 4),
        interpolation="bilinear"
    )(x)

    # Final 1x1 convolution with sigmoid activation for binary prediction
    outputs = Conv2D(
        1,
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs, name="DeepLabV3Plus")
    return model
