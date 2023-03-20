from __future__ import print_function
# coding: utf-8

######################################################################################
# # TRAIN:
import os
import cv2
# simplified interface for building models
import keras
import pickle
import numpy as np
import variables as vars
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import keras
import time

# from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import optimizers
# from tensorflow.keras.regularizers import l2
# from imutils import paths
import argparse

# because our models are simple
from keras.models import Sequential
from keras.models import model_from_json
# for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from scipy.misc import imread, imresize, imshow
# dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
# into respective layers
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array

# from tensorflow.keras.layers import Input, Dense, BatchNormalization
# from IPython.core.display import Image

def preprocessing_img(x):
    # resize + binarization
    x = imresize(x, (img_rows, img_cols))
    __, x = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)

    # dilate
    morph_size = (2, 2)
    cpy = x.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    x = ~cpy
    return x

def build_model(model, input_shape):
    # convolutional layer with rectified linear unit activation
    # 1
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))    
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 5
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 6
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 7
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 8
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    # fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    # fully connected to get all relevant data
    model.add(Dense(num_classes, activation='relu'))
    # one more dropout for convergence' sake :)
    model.add(Dropout(0.5))
    # output a softmax to squash the matrix into output probabilities
    model.add(Dense(num_classes, activation='softmax'))


    # # convolutional layer with rectified linear unit activation
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # # again
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # # choose the best features via pooling
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # randomly turn neurons on and off to improve convergence
    # model.add(Dropout(0.25))
    # # flatten since too many dimensions, we only want a classification output
    # model.add(Flatten())
    # # fully connected to get all relevant data
    # model.add(Dense(128, activation='relu'))
    # # one more dropout for convergence' sake :)
    # model.add(Dropout(0.5))
    # # output a softmax to squash the matrix into output probabilities
    # model.add(Dense(num_classes, activation='softmax'))

    print("[INFO] compiling model...")
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])   # cpu
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])   # gpu
    return model

# def plot_filters(model, layer_ith, n_filters):
    # # filters in the model
    # layer = model.layers
    # # print('*' * 9)
    # # print(len(layer))  # 8 layers
    # filters, biases = model.layers[layer_ith].get_weights()
    # print(layer[layer_ith].name, filters.shape)

    #
    # # plot filters
    # fig1 = plt.figure(figsize=(8, 12))
    # for i in range(1, n_filters + 1):
    #     f = filters[:, :, :, i - 1]
    #     fig1 = plt.subplot(rows, columns, i)
    #     fig1.set_xticks([])  # turn off axis
    #     fig1.set_yticks([])
    #     plt.imshow(f[:, :, layer_ith], cmap='gray')  # show only the filters from 0th channel
    # plt.show()

def plot_filters(model):
    for layer_ith in conv_layer_index:
        filters, biases = model.layers[layer_ith].get_weights()
        # print('^'*9)
        # # print(len(layer))  # 8 layers
        # print(layer[layer_ith].name, filters.shape)
        # # layer[0]: conv2d (3, 3, 1, 32)
        # # layer[1]: conv2d_1 (3, 3, 32, 64)
        # print(model.layers[layer_ith])  # <keras.layers.convolutional.Conv2D object at 0x7f22deeb2b38>
        # print(np.array(filters).shape)  # model.layers[0]:(3, 3, 1, 32), model.layers[1]:(3, 3, 32, 64)
        # print(filters[:, :, layer_ith, 0])
        # time.sleep(10000)
        n_filters = model.layers[layer_ith].filters
        rows = n_filters / columns
        fig1 = plt.figure(figsize=(8, 12))
        for i in range(1, n_filters + 1):
            fig1 = plt.subplot(rows, columns, i)
            fig1.set_xticks([])  # turn off axis
            fig1.set_yticks([])
            plt.imshow(filters[:, :, layer_ith, i-1], cmap='gray')  # show only the filters from 0th channel
        plt.show()

# def plot_features(model):
#     # plot filter outputs
#     outputs = [model.layers[i].output for i in conv_layer_index]
#     # from keras.models import Model
#     model_short = Model(inputs=model.inputs, outputs=outputs)
#     print(model_short.summary())
#
#     # from keras.preprocessing.image import load_img, img_to_array
#     # img = load_img('./predict_this_doc/1.jpg', target_size=(img_rows, img_cols))
#     img = cv2.imread('./predict_this_doc/1.jpg', cv2.IMREAD_GRAYSCALE)
#     img = preprocessing_img(img)  # preprocessing
#
#     # convert the image to an array
#     img = img_to_array(img)
#
#     # expand dimensions to match the shape of model input
#     img = np.expand_dims(img, axis=0)
#     # img = np.expand_dims(img, axis=2)
#
#     # generate feature output by predicting on the input image
#     feature_output = model_short.predict(img)
#     # print('&'*9)
#     # print(len(feature_output))
#
#     layer_ith = 0
#
#     for ftr in feature_output:
#         fig2 = plt.figure(figsize=(12, 12))
#         n_filters = model.layers[layer_ith].filters
#         rows = n_filters / columns
#         for i in range(1, n_filters + 1):
#             fig2 = plt.subplot(rows, columns, i)
#             fig2.set_xticks([])
#             fig2.set_yticks([])
#             plt.imshow(ftr[layer_ith, :, :, i-1], cmap='gray')
#         plt.show()

def plot_features(model):
    # plot filter outputs
    outputs = [model.layers[i].output for i in conv_layer_index]
    # print('&'*9)
    # print(outputs)  # [<KerasTensor: shape=(None, 254, 254, 32) dtype=float32 (created by layer 'conv2d')>, <KerasTensor: shape=(None, 252, 252, 64) dtype=float32 (created by layer 'conv2d_1')>]
    # print(outputs[0])  # KerasTensor(type_spec=TensorSpec(shape=(None, 254, 254, 32), dtype=tf.float32, name=None), name='conv2d/Relu:0', description="created by layer 'conv2d'")
    # print(model.inputs)  # [<KerasTensor: shape=(None, 256, 256, 1) dtype=float32 (created by layer 'conv2d_input')>]
    # time.sleep(10000)
    # from keras.models import Model
    model_short = Model(inputs=model.inputs, outputs=outputs)
    print(model_short.summary())

    # from keras.preprocessing.image import load_img, img_to_array
    # img = load_img('./predict_this_doc/1.jpg', target_size=(img_rows, img_cols))
    # img = cv2.imread('./predict_this_doc/1.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('./19_75.jpg', cv2.IMREAD_GRAYSCALE)
    img = preprocessing_img(img)  # preprocessing
    cv2.imshow('preprocessing19', img)
    cv2.imwrite('preprocessing19.png', img)
    # time.sleep(10000)

    # convert the image to an array
    img = img_to_array(img)

    # expand dimensions to match the shape of model input
    img = np.expand_dims(img, axis=0)
    # img = np.expand_dims(img, axis=2)

    # generate feature output by predicting on the input image
    feature_output = model_short.predict(img)
    # print('&'*9)
    # print(feature_output)  # array value
    # print(np.array(feature_output[0]).shape)  # (1, 254, 254, 32)
    # print(np.array(feature_output[1]).shape)  # (1, 254, 254, 64)
    # time.sleep(10000)

    # for ftr in feature_output:
    for layer_ith in conv_layer_index:
        n_filters = model.layers[layer_ith].filters
        rows = n_filters / columns
        # print('%'*9)
        # print(n_filters)
        # print(rows)
        # print(columns)
        fig2 = plt.figure(figsize=(12, 12))
        for i in range(1, n_filters + 1):
            # print('*'*9)
            # print(i)
            # time.sleep(10000)
            f1 = feature_output[layer_ith]
            # print('%'*9)
            # print(np.array(f1).shape)  # feature_output[0]:(1, 254, 254, 32), feature_output[1]:(1, 254, 254, 64)
            fig2 = plt.subplot(rows, columns, i)
            fig2.set_xticks([])
            fig2.set_yticks([])
            plt.imshow(f1[0, :, :, i-1], cmap='gray')
        plt.show()

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

    model.save('model_01.h5')
    print("Saved model to disk")

if __name__ == '__main__':

    # # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # TF >= 2.0
    # print("Num GPUs Available: ", len(tf.compat.v1.config.experimental.list_physical_devices('GPU')))  # TF 1.15 or below
    # import sys
    # sys.exit()

    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'   # gpu

    # # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-d", "--dataset", required=True,
    # 	help="path to input dataset")
    # ap.add_argument("-e", "--epochs", type=int, default=50,
    # 	help="# of epochs to train our network for")
    # ap.add_argument("-p", "--plot", type=str, default="plot.png",
    # 	help="path to output loss/accuracy plot")
    # args = vars(ap.parse_args())

    img_rows, img_cols = vars.img_rows, vars.img_cols
    batch_size = vars.batch_size
    num_classes = vars.num_classes
    epochs = vars.epochs
    model_json_path = vars.model_json_path
    model_path = vars.model_path
    prediction_file_dir_path = vars.prediction_file_dir_path

    path = '../../data/2classes_nosplit_balanced/eg2/'

    # # data preparation
    data = []
    labels = []

    for folder, subfolders, files in os.walk(path):
      for name in files:
        if name.endswith('.jpg'):
          x = cv2.imread(folder + '/' + name, cv2.IMREAD_GRAYSCALE)
          # cv2.imshow('original', x)
          # print("$"*9)
          # print(x.dtype)  # uint8
          # print(x.shape)  # (6600, 5100)
          x = preprocessing_img(x)  # preprocessing
          # cv2.imshow('preprocessing', x)
          # cv2.imwrite('preprocessing_x.png', x)
          # print("!" * 9)
          # print(x.dtype)  # uint8
          # print(x.shape)  # (256, 256)

          # x = np.expand_dims(x, axis=4)
          x = np.expand_dims(x, axis=2)
          # cv2.imshow('expand_dims', x)
          # cv2.imwrite('expand_dim_x.png', x)
          # print("@" * 9)
          # print(x.dtype)  # uint8
          # print(x.shape)  # (256, 256, 1)

          data.append(x)
          # cv2.imshow('data_x', x)
          # cv2.imwrite('data_x.png', x)
          # print("#" * 9)
          # print(x.dtype)  # uint8
          # print(x.shape)  # (256, 256, 1)

          # # pause
          # time.sleep(100000)

          # cv2.imwrite(str(name) + '00986.jpg', x)
          labels.append(os.path.basename(folder))

    data1 = np.asarray(data)
    labels1 = np.asarray(labels)

    # # perform one-hot encoding on the labels
    # lb = LabelBinarizer()
    # labels = lb.fit_transform(labels)

    # x_train, x_test, y_train, y_test = train_test_split(data1, labels1,
    #                                                     random_state=0,
    #                                                     test_size=0.25
    #                                                     )
    # # x_train, x_test, y_train, y_test = train_test_split(data1, labels1,
    #                                                     random_state=0,
    #                                                     test_size=0.4
    #                                                     )
    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
    x = data1
    y = labels1
    for train_index, test_index in skf.split(x, y):
        print("TRAIN:", train_index, '\n', "TEST:", test_index, '\n')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('y_train:', y_train, '\n', 'y_test:', y_test, '\n')

    x_test_all = x_test
    y_test_all = y_test
    print('!' * 9)
    print(int(len(x_test)))  # 90=(150+75)*0.4
    x_test = x_test_all[:int(len(x_test_all)/2)]  # validation_x, train_val_test=6:2:2
    y_test = y_test_all[:int(len(y_test_all)/2)]  # validation_y
    print('*'*9)
    print(int(len(x_test_all)/2))  # 45
    print('y_val:', y_test, '\n')

    x_test1 = x_test_all[int(len(x_test_all)/2)+1:]  # test_x
    y_test1 = y_test_all[int(len(y_test_all)/2)+1:]  # test_y
    print('@'*9)
    print(int(len(x_test1)))  # 44
    print(int(len(x_test_all)/2)+1)  # 46
    print('y_test1', y_test1, '\n')
    # time.sleep(10000)

    # convert the data into a NumPy array
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_test1 = np.array(x_test1)
    y_test1 = np.array(y_test1)

    # preprocess by scaling all pixel intensities to the range [0, 1]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    x_test1 = x_test1.astype('float32')
    x_test1 /= 255
    # data = np.array(data, dtype="float") / 255.0

    # # build our model
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model = build_model(model, input_shape)

    # print model info
    model.summary()
    # print(model.summary())
    # save the model as a graph
    tf.keras.utils.plot_model(model, "NNStr.png", show_shapes=True)

    # # train the model
    # print("[INFO] training network for {} epochs...".format(args["epochs"]))
    # train that ish!
    lb = LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    y_test1 = lb.fit_transform(y_test1)
    # # perform one-hot encoding on the labels
    # lb = LabelBinarizer()
    # labels = lb.fit_transform(labels)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    y_test1 = tf.keras.utils.to_categorical(y_test1)

    # # with augmentation
    # # construct the training image generator for data augmentation
    # aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    # 	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    # 	horizontal_flip=True, fill_mode="nearest")
    # x = aug.flow(x_train, y_train, batch_size=batch_size)
    # # model_fit_output = model.fit(x,
    # #                              epochs=epochs,
    # #                              verbose=1,
    # #                              validation_data=(x_test, y_test))  # cpu
    # from math import ceil
    # n_points = len(x_train)
    # batch_size = 8
    # steps_per_epoch = ceil(n_points / batch_size)
    #
    # model_fit_output = model.fit_generator(x,
    #                              epochs=epochs,
    #                              verbose=1,
    #                              validation_data=(x_test, y_test))  # gpu

    # without augmentation
    model_fit_output = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))  # cpu
    # model_fit_output = model.fit_generator(x_train, y_train,
    #                              batch_size=batch_size,
    #                              epochs=epochs,
    #                              verbose=1,
    #                              validation_data=(x_test, y_test))  # gpu

    # # evaluation
    # score = model.evaluate(x_test, y_test, verbose=0)
    score = model.evaluate(x_test1, y_test1, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # evaluate the network
    # out = model.predict(x_test)
    # print("[INFO] evaluating network...")
    # print(classification_report(y_test.argmax(axis=1),
    #                             out.argmax(axis=1), target_names=lb.classes_))
    out = model.predict(x_test1)
    print("[INFO] evaluating network...")
    print(classification_report(y_test1.argmax(axis=1),
                                out.argmax(axis=1), target_names=lb.classes_))

    # plot training loss and accuracy
    # N = args["epochs"]
    plot_loss_accuracy(model_fit_output, epochs)

    # # Save the model
    save_model(model)

    # # # visualization
    # # draw filters
    # columns = 8
    # conv_layer_index = [0, 1]  # according to your network structure
    # plot_filters(model)
    # # time.sleep(10000)
    #
    # # draw features
    # plot_features(model)
    # # time.sleep(10000)