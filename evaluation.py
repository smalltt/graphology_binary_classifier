# This script is for model evaluation

import tensorflow as tf
from tensorflow import keras

def gen_test_ds(arg_ds_dir, arg_image_size, arg_batch_size):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=arg_ds_dir,  
        batch_size=arg_batch_size,
        image_size=(arg_image_size, arg_image_size),
        seed=123, # seed for what?
        )

    return test_ds

# constant
input_im_size = (150, 150)
model_default_save_path  = 'model_files/tf_model.save'

test_ds_dir = "/usr/test/eg1_mnist/testing"
IMAGE_SIZE = 150
BATCH_SIZE = 32

# load trained model
loaded_model = tf.keras.models.load_model(model_default_save_path)
loaded_model.summary()

# generate testing dataset
test_ds = gen_test_ds(test_ds_dir,IMAGE_SIZE,BATCH_SIZE)

# run model evaluation
score = loaded_model.evaluate(test_ds, verbose=1)  
print('Test loss:', score[0])
print('Test accuracy:', score[1])
