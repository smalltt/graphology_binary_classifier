# train own dataset

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
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

## Plot
def plot_loss_accuracy(model_fit_output, n_epochs):
    plt.style.use("ggplot")
    plt.figure()
    H = model_fit_output
    N = n_epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["binary_accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_binary_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    plt.savefig('loss_acc.png', bbox_inches='tight')
    plt.show()

## Save model function
def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)

    # pickle label encoder obj
    with open(label_obj_path, 'wb') as lb_obj:
        pickle.dump(lb, lb_obj)

    # serialize weights to HDF5
    model.save_weights(model_path)

    model.save('model_train.h5')
    print("Finish saving training model")


## Variables
IMAGE_SIZE = 150
INPUT_SHAPE = (150, 150, 3)
size = (150, 150)
BATCH_SIZE = 32  # 32
model_json_path = 'model_files/model.json'
label_obj_path = 'model_files/labels.sav'
model_path = 'model_files/model.h5'
lb = LabelEncoder()
print('%'*9)
print('lb = ',lb)  # lb =  LabelEncoder()

epochs = 100  # 25, 50, 100, 2000, 5000

## Split samples as training and validation datasets
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/usr/test/data/2classes_split_balanced/eg1/split_9_12/training",  # 2classes_nosplit_balanced/eg1, 2classes_split_balanced/eg2/split_6_8
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="both",
)   # Found 128 files belonging to 2 classes.
    # Using 103 files for training.
    # Using 25 files for validation.


test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/usr/test/data/2classes_split_balanced/eg1/split_9_12/testing",  
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    seed=123,
)  # Found 20 files belonging to 2 classes.

# get label from test_ds
for y_test_image, y_test_label in test_ds.take(1):
    print('='*9)
    print('y_test_image shape: ', y_test_image.numpy().shape)
    print('y_test_label: ', y_test_label.numpy())  # tf.Tensor([1 1 0 0 1 0 0 1 1 1 1 0 1 1 0 1 0 0 0 0], shape=(20,), dtype=int32)
    # y_test.append(y_test_label)  #?

print('^'*9)
print(y_test_label)  # tf.Tensor([1 1 0 0 1 0 0 1 1 1 1 0 1 1 0 1 0 0 0 0], shape=(20,), dtype=int32)
print(y_test_label.numpy())  # [1 1 0 0 1 0 0 1 1 1 1 0 1 1 0 1 0 0 0 0]
y_test = y_test_label.numpy()


class_names = train_ds.class_names
num_classes = len(class_names)
print('~'*9)
print(class_names)  # ['conscientiousness', 'extraversion']

print('#'*9)
print('Finish preparing dataset!!!')


## Visualize first 9 images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])  ## 0: conscientiousness; 1: extraversion     
        plt.axis("off")
        plt.savefig('first9images.png')
print('*'*9)
print('labels = ', labels)
print('Finish visualizing first 9 images!!!')


## Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),  
    layers.RandomRotation(0.1), 
    # layers.RandomTranslation(height_factor=(-0.2, 0.3)),  # has problem???
    layers.RandomContrast(factor=[0.1, 0.9]),
    layers.Rescaling(scale=1 / 255)  # [0, 1]
    # layers.Rescaling(scale=1 / 127.5, offset=-1),  #[-1, 1]   
])

# # Visualize augmented samples  # has problem???
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)        
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.title(class_names[labels[0]])  
#         plt.axis("off")
#         plt.savefig('0th_augmented_image.png')
# print('*'*9)
# print('Finish visualizing augmented images!!!')


## Built a model
# base_model = keras.applications.Xception(  # Xception and imagenet relationship???
#     weights="imagenet",  # Load weights pre-trained on ImageNet.
#     input_shape=INPUT_SHAPE,
#     include_top=False,
# )  # Do not include the ImageNet classifier at the top.
base_model = keras.applications.ConvNeXtTiny(
    model_name="convnext_tiny",
    include_top=False,  # Whether to include the fully-connected layer at the top of the network. Defaults to True.
    include_preprocessing=True,  # ??? https://www.tensorflow.org/guide/keras/preprocessing_layers
    weights="imagenet",  # pre-training on ImageNet-1k
    input_tensor=None,
    input_shape=INPUT_SHAPE,  # Optional shape tuple, only to be specified if include_top is False. It should have exactly 3 inputs channels. 
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)
# For ConvNeXt, preprocessing is included in the model using a Normalization layer. 

## Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=INPUT_SHAPE)
x = data_augmentation(inputs)  # Apply random data augmentation ??? https://www.tensorflow.org/guide/keras/preprocessing_layers

# ConvNeXt models expect their inputs to be float or uint8 tensors of pixels with values in the [0-255] range. 
# # the rescaling layer outputs: `(inputs * scale) + offset`
# scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)  # [-1, 1]
# x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
## Model
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_2 (InputLayer)        [(None, 150, 150, 3)]     0         
                                                                 
#  sequential (Sequential)     (None, None, None, 3)     0         
                                                                 
#  rescaling (Rescaling)       (None, 150, 150, 3)       0         
                                                                 
#  convnext_tiny (Functional)  (None, 4, 4, 768)         27820128  
                                                                 
#  global_average_pooling2d (G  (None, 768)              0         
#  lobalAveragePooling2D)                                          
                                                                 
#  dropout (Dropout)           (None, 768)               0         
                                                                 
#  dense (Dense)               (None, 1)                 769       
                                                                 
# =================================================================
# Total params: 27,820,897
# Trainable params: 769
# Non-trainable params: 27,820,128
# _________________________________________________________________

# keras.utils.plot_model(model, show_shapes=True)
print('$'*9)
print('Finish building model!!!')


## Train the top layer
# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
# ]
callbacks=[
    #给tensorboard仪表盘提供数据, how to use???
    tf.keras.callbacks.TensorBoard(log_dir='./tf_cnn_board', histogram_freq=1),
    #执行过程中动态调整LearningRate。 本例是初始0.001，每10轮lr减半???
    # tf.keras.callbacks.LearningRateScheduler(lambda epoch:0.001/np.power(2,int(epoch/10))),
    # tf.keras.callbacks.LearningRateScheduler(lambda epoch:0.001/np.power(2,int(epoch/10))),  ## training samples different from my own dataset
    # #提前终止条件。本例是验证集的预测精度超过5轮都没有提升了，就终止???
    # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.0001,patience=5,restore_best_weights=True),
    # #每训练一轮，就把当前的模型及参数保存一次（通过save_freq控制每轮保存次数）
    # tf.keras.callbacks.ModelCheckpoint(filepath="tf_model_epoch-{epoch:04d}", verbose=1, save_weights_only=False, save_freq=50)
]

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)  # default lr=0.001; metrics=['accuracy']
# model.compile(   # ??? https://opendatascience.com/guidelines-for-choosing-an-optimizer-and-loss-functions-when-training-neural-networks/
#     optimizer=keras.optimizers.Adam(1e-3),   # ??? https://analyticsindiamag.com/guide-to-tensorflow-keras-optimizers/  
#     # Adam->Stochastic gradient descent(https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e)
#     loss="binary_crossentropy",  # ??? https://analyticsindiamag.com/ultimate-guide-to-loss-functions-in-tensorflow-keras-api-with-python-implementation/
#     metrics=["accuracy"]
# )
# model.compile(
#     loss=keras.losses.categorical_crossentropy,
#     optimizer=tf.keras.optimizers.Adadelta(),
#     metrics=['accuracy']
# )

## Maximize GPU utilization
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# # optimize loading speed
# batch_size = 32
# train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# validation_ds = val_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# # test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)


## Resize images to 150x150 (If too big, will have errors)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))


## Train
model_fit_output = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
print('@'*9)
print('Finish training!!!')

## Save model
save_model(model)  ## save_model
model.save('tf_model.save')  ## model.save
print('+'*9)
print('Finish saving model!!!')

## evaluation
# plot training loss and accuracy
# N = args["epochs"]
plot_loss_accuracy(model_fit_output, epochs) #???how to save?

# score = model.evaluate(x_test, y_test, verbose=0)
score = model.evaluate(test_ds, verbose=1)  
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# 1/1 [==============================] - 2s 2s/step - loss: 0.7398 - accuracy: 0.5000
# Test loss: 0.7398015856742859
# Test accuracy: 0.5


# evaluate the network
# out = model.predict(x_test)
# print("[INFO] evaluating network...")
# print(classification_report(y_test.argmax(axis=1),
#                             out.argmax(axis=1), target_names=lb.classes_))
out = model.predict(test_ds)
print('!'*9)
print(out.argmax(axis=1))  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
print(y_test)
print("[INFO] evaluating network...")
print(classification_report(y_test, out.argmax(axis=1)))
#               precision    recall  f1-score   support

#            0       0.50      1.00      0.67        10
#            1       0.00      0.00      0.00        10

#     accuracy                           0.50        20
#    macro avg       0.25      0.50      0.33        20
# weighted avg       0.25      0.50      0.33        20


# ## Do a round of fine-tuning of the entire model
# # Unfreeze the base_model. Note that it keeps running in inference mode
# # since we passed `training=False` when calling it. This means that
# # the batchnorm layers will not update their batch statistics.
# # This prevents the batchnorm layers from undoing all the training
# # we've done so far.
# base_model.trainable = True
# model.summary()


# model.compile(
#     optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
#     loss=keras.losses.BinaryCrossentropy(from_logits=True),
#     metrics=[keras.metrics.BinaryAccuracy()]
# )

# epochs = 10  # 5
# model.fit(
#     train_ds, 
#     epochs=epochs, 
#     callbacks=callbacks,
#     validation_data=val_ds
# )
# print('*'*9)
# print('Finish fine-tuning!!!')

# ## save model
# # model.save('tf_model_fineTun')
# save_model(model)
# model.save('model.saveFT')
# print('*'*9)
# print('Finish saving fine-tuning model!!!')


# # pause
# import time
# time.sleep(10000)