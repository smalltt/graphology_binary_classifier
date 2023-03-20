import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import plot_model


## clear warnings
tf.get_logger().setLevel('ERROR')  
# refer to: https://stackoverflow.com/questions/73304934/tensorflow-data-augmentation-gives-a-warning-using-a-while-loop-for-converting
# This seems to be a bug in keras version 2.9 and 2.10 (which is included in tensorflow): https://github.com/keras-team/keras-cv/issues/581
# It works correctly with TF v2.8.3 - no error messages, and training is fast.
# On my arch system – I have had installed TF by installing the python-tensorflow-opt-cuda package using pacman – I issued the following command which solved the issue:
# $ python -m pip install tensorflow-gpu==2.8.3


## Memory fragmentation
# import os
# os.environ['TF_GPU_ALLOCATOR'] ='cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

## Save model function
def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)

    # pickle label encoder obj
    with open(vars.label_obj_path, 'wb') as lb_obj:
        pickle.dump(lb, lb_obj)

    # serialize weights to HDF5
    model.save_weights(model_path)

    model.save('model_fineTun.h5')
    print("Finish saving fine tuning model!!!")

## Fine-tuning
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.

## Variables
INPUT_SHAPE = (150, 150, 3)
BATCH_SIZE = 32
IMAGE_SIZE =150
epochs = 2  # 5
model_json_path = 'model_files_fineTun/model_ft.json'
label_obj_path = 'model_files_fineTun/labels_ft.sav'
model_path = 'model_files_fineTun/model_ft.h5'
lb = LabelEncoder()

## Model preparation
base_model = keras.applications.ConvNeXtTiny(
    model_name="convnext_tiny",
    include_top=False,
    include_preprocessing=True,  ##???
    weights="imagenet",
    input_tensor=None,
    input_shape=INPUT_SHAPE,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)
base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()]
)


## Fine-tuning
# get: train_ds, val_ds
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/usr/test/data/2classes_nosplit_balanced/eg1",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    seed=1777,
    validation_split=0.2,
    subset="both",
)
# get: callbacks
callbacks=[
    #给tensorboard仪表盘提供数据, how to use???
    tf.keras.callbacks.TensorBoard(log_dir='./tf_cnn_board_fineTuning', histogram_freq=1),
    #执行过程中动态调整LearningRate。 本例是初始0.001，每10轮lr减半???
    tf.keras.callbacks.LearningRateScheduler(lambda epoch:0.001/np.power(2,int(epoch/10))),
    # #提前终止条件。本例是验证集的预测精度超过5轮都没有提升了，就终止???
    # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.0001,patience=5,restore_best_weights=True),
    # #每训练一轮，就把当前的模型及参数保存一次（通过save_freq控制每轮保存次数）
    # tf.keras.callbacks.ModelCheckpoint(filepath="tf_model_epoch-{epoch:04d}", verbose=1, save_weights_only=False, save_freq=50)
]

## Load model
model = tf.keras.models.load_model('tf_model.save')

## Maximize GPU utilization
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

## Resize images to 150x150 (If too big, will have errors)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

## Fine-tuning
model.fit(
    train_ds, 
    epochs=epochs, 
    callbacks=callbacks,
    validation_data=val_ds
)
print('*'*9)
print('Finish fine-tuning!!!')

## Save model
model.save('tf_model_fineTun.save')
save_model(model)
print('*'*9)
print('Finish saving fine-tuning model!!!')