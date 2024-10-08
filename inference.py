# This script is for model inference/prediction

import tensorflow as tf
from tensorflow import keras
import conf

tf.get_logger().setLevel('ERROR')  # clear warnings

if __name__=="__main__":

    # load trained model
    loaded_model = tf.keras.models.load_model(conf.epoch_chosen)
    loaded_model.summary()

    # load image for inference
    img = keras.preprocessing.image.load_img(
        conf.pred_img_path, target_size=conf.input_im_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # run inference
    predictions = loaded_model.predict(img_array)
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% 0(conscientiousness) and {100 * score:.2f}% 1(extraversion).")