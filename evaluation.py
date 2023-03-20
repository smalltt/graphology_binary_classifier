# This script is for model evaluation

import tensorflow as tf
from tensorflow import keras
import conf
from common_utils import dataset as ds
from sklearn.metrics import classification_report

if __name__=="__main__":
    # load trained model
    loaded_model = tf.keras.models.load_model(conf.model_best_file_path)
    loaded_model.summary()

    # generate testing dataset
    test_ds = ds.gen_test_ds(conf.test_ds_dir,conf.image_size,conf.batch_size)

    # run model evaluation
    score = loaded_model.evaluate(test_ds, verbose=1)  
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # get label from test_ds
    y_test = []
    for y_test_image, y_test_label in test_ds.take(-1):
        print('='*9)
        print('y_test_image shape: ', y_test_image.numpy().shape)
        print('y_test_label: ', y_test_label)  # tf.Tensor([1 1 0 0 1 0 0 1 1 1 1 0 1 1 0 1 0 0 0 0], shape=(20,), dtype=int32)
        print('y_test_label.numpy: ', y_test_label.numpy())
        y_test = y_test + y_test_label.numpy().tolist()
        print('y_test', y_test)
    
    # evaluate the network
    out = loaded_model.predict(test_ds)
    print('!'*9)
    print('out.argmax(axis=1): ', out.argmax(axis=1))  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    print("[INFO] evaluating network...")
    print(classification_report(y_test, out.argmax(axis=1)))