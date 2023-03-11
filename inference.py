# This script is for model inference/prediction

import tensorflow as tf
from tensorflow import keras

# constant
input_im_size = (150, 150)
model_default_save_path  = 'model_files/tf_model.save'

# load trained model
loaded_model = tf.keras.models.load_model(model_default_save_path)
loaded_model.summary()

# load image for inference
img = keras.preprocessing.image.load_img(
    "eg1_mnist/testing/0/3.png", target_size=input_im_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# run inference
predictions = loaded_model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% 0 and {100 * score:.2f}% 1.")
