# refer ChatGPT

# Library
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import plot_model



# Variables
img_dir="/usr/test/data/2classes_nosplit_balanced/eg1"  # 2classes_nosplit_balanced/eg1, 2classes_split_balanced/eg2/split_6_8
IMAGE_SIZE=(256, 256)
BATCH_SIZE=32
INPUT_SHAPE = (150, 150, 3)
epoch=2
model_json_path = 'model_files/model.json'
label_obj_path = 'model_files/labels.sav'
model_path = 'model_files/model.h5'
lb = LabelEncoder()

## Functions
## Plot
def plot_loss_accuracy(model_fit_output, n_epochs):
    plt.style.use("ggplot")
    plt.figure()
    H = model_fit_output
    N = n_epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # plt.savefig(args["plot"])
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

    model.save('model_train_rf.h5')
    print("Finish saving training model")


## Create dataset

# get the total number of classes
class_names = os.listdir(img_dir)
# num_classes = len(class_names)

# # generate labels, one-hot encoded
# img_labels = [tf.keras.utils.to_categorical(
#     [i], num_classes=num_classes) for i in range(num_classes)]
print('*'*9)
print('class name=', class_names)
# print('num_classes=', num_classes)
# print('img_labels=', img_labels)


# # Shuffle the list of filenames
# np.random.shuffle(img_dir)
    
# # Split the set into training data, 80% of total images
# num_train = int(len(img_dir) * 0.8)
# train_filenames = img_dir[:num_train]

# # Validation data 10% of total images
# num_val = int(len(img_dir) * 0.1)
# val_filenames = img_dir[num_train:num_train + num_val]

# # Test data 10% of total images
# num_test = int(len(img_dir) * 0.1)
# test_filenames = img_dir[num_train + num_val:]



# Shuffle the list of filenames
# np.random.shuffle(img_dir)
# Split
X_train, X_test, y_train, y_test = train_test_split(img_dir, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# # shuffle(default) and split dataset into training, validation, and testing (80/10/10 respectively)
# train_ds, test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   img_dir,
#   labels="inferred",  # labels are generated from the directory structure
#   label_mode="categorical",  # labels are encoded as a categorical vector
#   class_names=class_names,
#   validation_split=0.2,
#   subset="test",
#   seed=123,
#   image_size=IMAGE_SIZE,
#   batch_size=BATCH_SIZE,
#   )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   train_ds,
#   labels="inferred",
#   label_mode="categorical",
#   class_names=class_names,
#   validation_split=0.1,
#   subset="validation",
#   seed=123,
#   image_size=IMAGE_SIZE,
#   batch_size=BATCH_SIZE,
#   )
# train_ds = train_ds-val_ds

# print the number of samples
print('+'*9)
print("Number of samples in Training Dataset:", len(X_train))
print("Number of samples in Validation Dataset: ", len(X_val))
print("Number of samples in Testing Dataset:", len(X_test))
class_names = train_ds.class_names
num_classes = len(class_names)
print('class_names=', class_names)
print('num_classes=', num_classes)
print('Finish preparing dataset!!!')


## Built a model
base_model = keras.applications.ConvNeXtTiny(
    model_name="convnext_tiny",
    include_top=False,  # Do not include the ImageNet classifier at the top
    include_preprocessing=True,  ##???
    weights="imagenet",  # Load weights pre-trained on ImageNet
    input_tensor=None,
    input_shape=INPUT_SHAPE,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)

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

## Train the top layer
callbacks=[
    #给tensorboard仪表盘提供数据, how to use???
    tf.keras.callbacks.TensorBoard(log_dir='./tf_cnn_board', histogram_freq=1),
    #执行过程中动态调整LearningRate。 本例是初始0.001，每10轮lr减半???
    # tf.keras.callbacks.LearningRateScheduler(lambda epoch:0.001/np.power(2,int(epoch/10))),
    tf.keras.callbacks.LearningRateScheduler(lambda epoch:0.001/np.power(2,int(epoch/10))),  ## training samples different from my own dataset
    # #提前终止条件。本例是验证集的预测精度超过5轮都没有提升了，就终止???
    # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.0001,patience=5,restore_best_weights=True),
    # #每训练一轮，就把当前的模型及参数保存一次（通过save_freq控制每轮保存次数）
    # tf.keras.callbacks.ModelCheckpoint(filepath="tf_model_epoch-{epoch:04d}", verbose=1, save_weights_only=False, save_freq=50)
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
# model.compile(
#     loss=keras.losses.categorical_crossentropy,
#     optimizer=tf.keras.optimizers.Adadelta(),
#     metrics=['accuracy']
# )

train_ds = (X_train, y_train)
val_ds = (X_val, y_val)
test_ds = (X_test, y_test)

# Maximize GPU utilization
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Resize images to 150x150 (If too big, will have errors)
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
print('*'*9)
print('Finish training!!!')
# record of the loss values and metric values during training
history.history

## Save model
save_model(model)  ## save_model
model.save('tf_model_rf.save')  ## model.save
print('*'*9)
print('Finish saving!!!')

## evaluation
# plot training loss and accuracy
# N = args["epochs"]
plot_loss_accuracy(model_fit_output, epochs)

# score = model.evaluate(x_test, y_test, verbose=0)
score = model.evaluate(test_ds, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# evaluate the network
# out = model.predict(x_test)
# print("[INFO] evaluating network...")
# print(classification_report(y_test.argmax(axis=1),
#                             out.argmax(axis=1), target_names=lb.classes_))
out = model.predict(test_ds[0])
print("[INFO] evaluating network...")
print(classification_report(test_ds[1].argmax(axis=1),
                            out.argmax(axis=1), target_names=lb.classes_))

# # pause
# import time
# time.sleep(10000)