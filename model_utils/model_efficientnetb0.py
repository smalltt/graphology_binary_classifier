# This is the EfficientNetB0 model
# refer to: https://keras.io/api/applications/efficientnet/#efficientnetb0-function

from tensorflow import keras
from tensorflow.keras import layers

def make_model(arg_input_shape, arg_num_classes):

    # base model
    base_model = keras.applications.EfficientNetB0(
    # model_name="EfficientNetB0",
    include_top=False,  # Whether to include the fully-connected layer at the top of the network. Defaults to True.
    weights="imagenet",  # pre-training on ImageNet-1k
    # include_preprocessing=True, 
    input_tensor=None,
    input_shape=arg_input_shape,  # Optional shape tuple, only to be specified if include_top is False. It should have exactly 3 inputs channels.
    pooling=None,
    classes=arg_num_classes,
    classifier_activation="softmax",
    )
    # For EfficientNet, input preprocessing is included as part of the model (as a Rescaling layer), and thus tf.keras.applications.efficientnet.preprocess_input is actually a pass-through function. EfficientNet models expect their inputs to be float tensors of pixels with values in the [0-255] range.
    ## Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=arg_input_shape)

    ## Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),  
        # layers.RandomRotation(0.1), 
        # layers.RandomTranslation(height_factor=(-0.2, 0.3)),  # has problem???
        layers.RandomContrast(factor=[0.1, 0.9]),
        layers.Rescaling(scale=1 / 255)  # [0, 1]
        # layers.Rescaling(scale=1 / 127.5, offset=-1),  # [-1, 1]   
    ])
    x = inputs
    x = keras.applications.efficientnet.preprocess_input(x)  # pass-through function
    # x = data_augmentation(x)  # Apply random data augmentation ??? https://www.tensorflow.org/guide/keras/preprocessing_layers
    
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1, activation = "sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    
    return model