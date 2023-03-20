# train own dataset

# refer to: https://keras.io/examples/vision/image_classification_from_scratch/
# refer to: https://keras.io/guides/transfer_learning/#an-endtoend-example-finetuning-an-image-classification-model-on-a-cats-vs-dogs-dataset
# refer to: https://zhuanlan.zhihu.com/p/508232168

import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import plot_model

tf.get_logger().setLevel('ERROR')  # clear warnings
# refer to: https://stackoverflow.com/questions/73304934/tensorflow-data-augmentation-gives-a-warning-using-a-while-loop-for-converting
# This seems to be a bug in keras version 2.9 and 2.10 (which is included in tensorflow): https://github.com/keras-team/keras-cv/issues/581
# It works correctly with TF v2.8.3 - no error messages, and training is fast.
# On my arch system – I have had installed TF by installing the python-tensorflow-opt-cuda package using pacman – I issued the following command which solved the issue:
# $ python -m pip install tensorflow-gpu==2.8.3


# tfds.disable_progress_bar()

IMAGE_SIZE = 256
BATCH_SIZE = 32

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/usr/test/data/2classes_nosplit_balanced/eg1",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    seed=1777,
    validation_split=0.2,
    subset="both",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)
# Found 148 files belonging to 2 classes.
# Using 119 files for training.
# Using 29 files for validation.
# ['conscientiousness', 'extraversion']
print('*'*9)
print('Finish input your own dataset!!!')


# visualize the first 9 images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))       
        plt.axis("off")
        plt.savefig('first9images.png')
print('*'*9)
print('Finish visualize first 9 images!!!')


# Data augmentation
data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)

# Visualize augmented samples
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)        
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.title(int(labels[0]))  
        plt.axis("off")
        plt.savefig('augmented_image0.png')
print('*'*9)
print('Finish visualize augmented images!!!')


## Built a model
INPUT_SHAPE = (150, 150, 3)
base_model = keras.applications.Xception(  # Xception and imagenet relationship???
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=INPUT_SHAPE,
    include_top=False,
)  # Do not include the ImageNet classifier at the top.


# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=INPUT_SHAPE)
x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be scaled from (0, 255) to a range of (-1., +1.) 
# the rescaling layer outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
# keras.utils.plot_model(model, show_shapes=True)
print('*'*9)
print('Finish building model!!!')

# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_2 (InputLayer)        [(None, 150, 150, 3)]     0         
                                                                 
#  sequential (Sequential)     (None, None, None, 3)     0         
                                                                 
#  rescaling (Rescaling)       (None, 150, 150, 3)       0         
                                                                 
#  xception (Functional)       (None, 5, 5, 2048)        20861480  
                                                                 
#  global_average_pooling2d (G  (None, 2048)             0         
#  lobalAveragePooling2D)                                          
                                                                 
#  dropout (Dropout)           (None, 2048)              0         
                                                                 
#  dense (Dense)               (None, 1)                 2049      
                                                                 
# =================================================================
# Total params: 20,863,529
# Trainable params: 2,049
# Non-trainable params: 20,861,480
# _________________________________________________________________


# Train the top layer
epochs = 2  # 25, 50
# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
# ]
callbacks=[
    #给tensorboard仪表盘提供数据, how to use???
    tf.keras.callbacks.TensorBoard(log_dir='./tf_cnn_board', histogram_freq=1),
    #执行过程中动态调整LearningRate。 本例是初始0.001，每10轮lr减半
    tf.keras.callbacks.LearningRateScheduler(lambda epoch:0.001/np.power(2,int(epoch/10))),
    #提前终止条件。本例是验证集的预测精度超过5轮都没有提升了，就终止
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.0001,patience=5,restore_best_weights=True),
    #每训练一轮，就把当前的模型及参数保存一次（通过save_freq控制每轮保存次数）
    tf.keras.callbacks.ModelCheckpoint(filepath="tf_model_epoch-{epoch:04d}", verbose=1, save_weights_only=False)
]

# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.BinaryCrossentropy(from_logits=True),
#     metrics=[keras.metrics.BinaryAccuracy()],
# )  # default lr=0.001; metrics=['accuracy']
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# # optimize loading speed
# batch_size = 32
# train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# validation_ds = val_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# # test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)


# resize images to 150x150
size = (150, 150)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
print('*'*9)
print('Finish training!!!')


## Do a round of fine-tuning of the entire model
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()]
)

epochs = 1  # 5
model.fit(train_ds, epochs=epochs, validation_data=val_ds)
print('*'*9)
print('Finish fine-tuning!!!')

## save model
model2.save('tf_model')
print('*'*9)
print('Finish saving model!!!')


## best model configuration, consider using KerasTuner
# refer to: https://github.com/keras-team/keras-tuner

# # pause
# import time
# time.sleep(10000)