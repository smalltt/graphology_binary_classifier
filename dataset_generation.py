# train own dataset

# refer to: https://keras.io/examples/vision/image_classification_from_scratch/
# refer to: https://keras.io/guides/transfer_learning/#an-endtoend-example-finetuning-an-image-classification-model-on-a-cats-vs-dogs-dataset
# refer to: https://zhuanlan.zhihu.com/p/508232168

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report
# import pickle

# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.utils import image_dataset_from_directory
# from tensorflow.keras.utils import plot_model


## Variables
IMAGE_SIZE = 150
INPUT_SHAPE = (150, 150, 3)
size = (150, 150)
BATCH_SIZE = 32
model_json_path = 'model_files/model.json'
label_obj_path = 'model_files/labels.sav'
model_path = 'model_files/model.h5'
# lb = LabelEncoder()
# print('*'*9)
# print('lb = ',lb)
# *********
# lb =  LabelEncoder()
# Found 148 files belonging to 2 classes.
# Using 119 files for training.
# Using 29 files for validation.
epochs = 100  # 25, 50


## Split samples as training and validation datasets
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/usr/test/data/2classes_nosplit_balanced/eg1/training",  # 2classes_nosplit_balanced/eg1, 2classes_split_balanced/eg2/split_6_8
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    seed=1777,
    validation_split=0.2,
    subset="both",
)

## Split samples as training and validation datasets
test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/usr/test/data/2classes_nosplit_balanced/eg1/testing",  # 2classes_nosplit_balanced/eg1, 2classes_split_balanced/eg2/split_6_8
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    seed=1777,
    # validation_split=0,
    # subset="both",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)
# GPU:0 with 3846 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2060, pci bus id: 0000:01:00.0, compute capability: 7.5
# ['conscientiousness', 'extraversion']
print('*'*9)
print('Finish preparing dataset!!!')
