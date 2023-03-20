# refer to: https://debuggercafe.com/image-classification-using-tensorflow-on-custom-dataset/

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib
matplotlib.style.use('ggplot')

IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = '/usr/test/data/2classes_nosplit_balanced/eg1'  # 2classes_nosplit_balanced/eg1, 2classes_split_balanced/eg2/split_6_8
# VALID_DATA_DIR = 'input/validation/validation/'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_generator = datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    shuffle=True,
    target_size=IMAGE_SHAPE,
)
# valid_generator = datagen.flow_from_directory(
#     VALID_DATA_DIR,
#     shuffle=False,
#     target_size=IMAGE_SHAPE,
# )