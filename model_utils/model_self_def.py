# This is the first customized model

from tensorflow import keras
from tensorflow.keras import layers

def make_model(arg_input_shape, arg_num_classes):
    inputs = keras.Input(shape=arg_input_shape)

    # # Entry block for data augmentation, but failed
    # # Image augmentation layers are expecting inputs to be rank 3 (HWC) or 4D (NHWC) tensors. Got shape: (1, None, 150, 150, 3)
    # x = layers.RandomFlip("horizontal")(inputs),
    # x = layers.RandomRotation(0.1)(x),
    # x = layers.RandomContrast(factor=[0.1, 0.9])(x),
    
    # x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if arg_num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = arg_num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)
