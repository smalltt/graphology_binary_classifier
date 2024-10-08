# This is the xception model
# refer to: https://keras.io/api/applications/xception/

from tensorflow import keras
from tensorflow.keras import layers

def make_model(arg_input_shape, arg_num_classes):

    # base model
    base_model = keras.applications.Xception(
    # model_name="xception",
    include_top=False,  # Whether to include the fully-connected layer at the top of the network. Defaults to True.
    # include_preprocessing=True,  # ??? https://www.tensorflow.org/guide/keras/preprocessing_layers
    weights="imagenet",  # pre-training on ImageNet-1k
    input_tensor=None,
    input_shape=arg_input_shape,  # optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (299, 299, 3). It should have exactly 3 inputs channels, and width and height should be no smaller than 71. 
    pooling=None,
    classes=arg_num_classes,
    classifier_activation="softmax",
    )
    # For Xception, call tf.keras.applications.xception.preprocess_input on your inputs before passing them to the model. xception.preprocess_input will scale input pixels between -1 and 1 
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
        # layers.Rescaling(scale=1 / 127.5, offset=-1),  #[-1, 1]   
    ])
    x = inputs
    x = keras.applications.xception.preprocess_input(x)  #[-1, 1]
    # x = data_augmentation(x)  # Apply random data augmentation ??? https://www.tensorflow.org/guide/keras/preprocessing_layers
    
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1, activation = "sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    
    return model