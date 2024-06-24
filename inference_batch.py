# This script is for model inference/prediction

import tensorflow as tf
from tensorflow import keras
import conf
import os

tf.get_logger().setLevel('ERROR')  # clear warnings

def batch_inference(file_paths,img_name):
    acculate_score = []
    for file_path in file_paths:
        if file_path[-3:] == "jpg" or file_path[-3:] == "bmp" and img_name in file_path:
            print(file_path)
    
            # load image for inference
            img = keras.preprocessing.image.load_img(
                file_path, target_size=conf.input_im_size
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis

            # run inference
            predictions = loaded_model.predict(img_array)
            score = float(predictions[0])
            acculate_score.append(score)
            print(f"{file_path} is {100 * (1 - score):.2f}% 0(conscientiousness) and {100 * score:.2f}% 1(extraversion).")

        else:
            continue
    
    average_score = sum(acculate_score)/len(acculate_score)
    print("="*10)
    print(f"Above image is {100 * (1 - average_score):.2f}% 0(conscientiousness) and {100 * average_score:.2f}% 1(extraversion).")

if __name__=="__main__":

    # load trained model
    loaded_model = tf.keras.models.load_model(conf.epoch_chosen)
    loaded_model.summary()

    img_dir_ls = ["split_1_1","split_2_1","split_3_4","split_6_8","split_9_12"]
    
    for img_dir_item in img_dir_ls:
        print("=>"*10)
        print(img_dir_item)
        
        img_dir = os.path.join('data/pregrocess_Bennie_Peleman/', img_dir_item)
        img_list = os.listdir(img_dir)
        file_paths = [os.path.join(img_dir, img_path) for img_path in img_list]

        img_name_ls = ["out3","out4-1","out4-2","out5-1","out5-2"]
        
        for img_name in img_name_ls:
            print("*-"*10)
            print(img_name)
            batch_inference(file_paths,img_name)

    # acculate_score = []
    # for file_path in file_paths:
    #     if file_path[-3:] == "jpg" or file_path[-3:] == "bmp" and "out3.bmp" in file_path:
    
    #         # load image for inference
    #         img = keras.preprocessing.image.load_img(
    #             file_path, target_size=conf.input_im_size
    #         )
    #         img_array = keras.preprocessing.image.img_to_array(img)
    #         img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    #         # run inference
    #         predictions = loaded_model.predict(img_array)
    #         score = float(predictions[0])
    #         acculate_score.append(score)
    #         print(f"{file_path} is {100 * (1 - score):.2f}% 0(conscientiousness) and {100 * score:.2f}% 1(extraversion).")

    #     else:
    #         continue
    
    # average_score = sum(acculate_score)/len(acculate_score)
    # print("="*10)
    # print(f"Above image is {100 * (1 - average_score):.2f}% 0(conscientiousness) and {100 * average_score:.2f}% 1(extraversion).")