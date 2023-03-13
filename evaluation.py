# This script is for model evaluation

import tensorflow as tf
from tensorflow import keras
import conf
from utils import dataset as ds

if __name__=="__main__":
    # load trained model
    loaded_model = tf.keras.models.load_model(conf.model_default_save_path)
    loaded_model.summary()

    # generate testing dataset
    test_ds = ds.gen_test_ds(conf.test_ds_dir,conf.image_size,conf.batch_size)

    # run model evaluation
    score = loaded_model.evaluate(test_ds, verbose=1)  
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
