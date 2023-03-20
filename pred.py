# from __future__ import print_function
# coding: utf-8

######################################################################################
# # TRAIN:
# import os
# import cv2
# # simplified interface for building models
# import keras
# import pickle
# import numpy as np
# import variables as vars
# import matplotlib.pyplot as plt
# # because our models are simple
# from keras.models import Sequential
# from keras.models import model_from_json
# # for convolution (images) and pooling is a technique to help choose the most relevant features in an image
# from keras.layers import Conv2D, MaxPooling2D
# from sklearn.preprocessing import LabelEncoder
# from keras.preprocessing.image import load_img, img_to_array
# from scipy.misc import imread, imresize, imshow
# # dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
# # into respective layers
# from keras.layers import Dense, Dropout, Flatten
# from sklearn.model_selection import train_test_split

# from keras.models import model_from_json

# img_rows, img_cols = vars.img_rows, vars.img_cols
# batch_size = vars.batch_size
# num_classes = vars.num_classes
# epochs = vars.epochs
# model_json_path = vars.model_json_path
# model_path = vars.model_path
# prediction_file_dir_path = vars.prediction_file_dir_path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle

from tensorflow import keras
# from keras.preprocessing.image import load_img, img_to_array
from keras.models import model_from_json
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

tf.get_logger().setLevel('ERROR')  # clear warnings

image_size = (150, 150)
img = keras.preprocessing.image.load_img(
    "1.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

model = tf.keras.models.load_model('tf_model.save')
# model2 = tf.keras.models.load_model('tf_model_fineTun')
# model.summary() # check model info

predictions = model.predict(img_array)
print('*'*9)
print(predictions)
# predictions2 = model2.predict(img_array)

###############################################################################
# PREDICTION
#############################################################################@##


# def print_results(class_lbl, out):
#   print('\n', '~' * 60)
#   for k, lbl in enumerate(class_lbl):
#     if lbl == 'LEFT_MARG':
#       print('\n > Courageous :', '\t' * 5, out[k] * 100, '%')
#       print('\n > Insecure and devotes oneself completely :\t',
#             100 - (out[k] * 100), '%')
#     elif lbl == 'RIGHT_MARG':
#       print('\n > Avoids future and a reserved person :\t', out[k] * 100, '%')
#     elif lbl == 'SLANT_ASC':
#       print('\n > Optimistic :', '\t' * 5, out[k] * 100, '%')
#     elif lbl == 'SLANT_DESC':
#       print('\n > Pessimistic :', '\t' * 4, out[k] * 100, '%')
#   print('~' * 60, '\n')

def print_results(class_lbl, out):
  print('\n', '~' * 60)
  for k, lbl in enumerate(class_lbl):
    if lbl == 'conscientiousness':
      print('\n > conscientiousness :', '\t' * 3, out[k] * 100, '%')
    elif lbl == 'extraversion':
      print('\n > extraversion :', '\t'* 3, out[k] * 100, '%')
  print('~' * 60, '\n')

def predict_personalities():

  json_file = open(model_json_path, 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  # from keras.models import model_from_json
  # loaded_model = model_from_json(
  #     open(
  #         model_json_path).read())
  loaded_model = model_from_json(loaded_model_json)  ###???
  # ValueError: Unknown layer: 'LayerScale'. Please ensure you are using a `keras.utils.custom_object_scope` and that this object is included in the scope. See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.

  # load weights into new model
  loaded_model.load_weights(model_path)
  print("*****Loaded Model from disk******")
  # try:
  #   json_file = open(model_json_path, 'r')
  #   loaded_model_json = json_file.read()
  #   json_file.close()

  #   # from keras.models import model_from_json
  #   # loaded_model = model_from_json(
  #   #     open(
  #   #         model_json_path).read())
  #   loaded_model = model_from_json(loaded_model_json)

  #   # load weights into new model
  #   loaded_model.load_weights(model_path)
  #   print("*****Loaded Model from disk******")
  # except Exception:
  #   return '\n\n> Need to train the model first!\n'
  # x = cv2.imread(prediction_file_dir_path + filename, cv2.IMREAD_GRAYSCALE)  # con
  x = cv2.imread('./1.jpg', cv2.IMREAD_GRAYSCALE)  # ext
  # x = imresize(x, (img_rows, img_cols))
  # __, x = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)

  # # dilate
  # morph_size = (2, 2)
  # cpy = x.copy()
  # struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
  # cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
  # x = ~cpy

  # convert the image to an array
  x = img_to_array(x)

  # x = np.expand_dims(x, axis=4)
  # x = np.expand_dims(x, axis=2)
  x = np.expand_dims(x, axis=0)
  out = loaded_model.predict(x, batch_size=16, verbose=0)

  with open(vars.label_obj_path, 'rb') as lb_obj:
    lb = pickle.load(lb_obj)

  # result = lb.inverse_transform(np.argmax(out[0]))
  print_results(lb.classes_, out[0])

  return '\n> Prediction Completed!'


if __name__ == '__main__':
  # fpath = None
  # for dir_0, sub_dir_0, files in os.walk(prediction_file_dir_path):
  #   fpath = files
  #   break
  # if fpath:
  #   res = predict_personalities(fpath[0])
  #   print(res)
  # else:
  #   print('No file found for prediction!')

  model_json_path = 'model_files/model.json'
  label_obj_path = 'model_files/labels.sav'
  model_path = 'model_files/model.h5'
  lb = LabelEncoder()
  res = predict_personalities()
  print(res)