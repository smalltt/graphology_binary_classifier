# prediction
# refer to: https://zhuanlan.zhihu.com/p/508232168

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

tf.get_logger().setLevel('ERROR')  # clear warnings

image_size = (150, 150)
img = keras.preprocessing.image.load_img("1.jpg", target_size=image_size)
# img = keras.preprocessing.image.load_img("19.jpg", target_size=image_size)
# img = keras.preprocessing.image.load_img("231.jpg", target_size=image_size)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

## Preprocessing
# data_augmentation = keras.Sequential(
#     [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
# )
# x = data_augmentation(img_array)
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(img_array)

model = tf.keras.models.load_model('tf_model.save')  # Train
# model = tf.keras.models.load_model('tf_model_fineTun')  # Fine-tuning
model.summary() # check model info

predictions = model.predict(x)
print('*'*9)
print(predictions)
# score = float(predictions[0])
# print(f"This image is {100 * (1 - score):.2f} %extraversion and {100 * score:.2f} %conscientiousness.")


# # load model and parameters
# model = tf.keras.models.load_model('tf_model_fineTun')
# model.summary() # check model info

# prediction
# IMAGE_SIZE = 150
# BATCH_SIZE = 32

# img = tf.keras.utils.load_img('1.jpg', target_size=(IMAGE_SIZE, IMAGE_SIZE))
# img_array = tf.keras.utils.img_to_array(img)
# predictions = model.predict(tf.expand_dims(img_array, 0))
# print('*'*9)
# print(predictions)  # [[0.11593238]]
# print(predictions[0])  # [0.11593238]


# img_index=np.argmax(predictions) # 2 
# train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
#     directory="/usr/test/data/2classes_nosplit_balanced/eg1",
#     batch_size=BATCH_SIZE,
#     image_size=(IMAGE_SIZE, IMAGE_SIZE),
#     shuffle=True,
#     seed=1777,
#     validation_split=0.2,
#     subset="both",
# )
# class_names = train_ds.class_names

# img_name=class_names[img_index] # roses
# print('*'*9)
# print(predictions, img_index, img_name)
# # [[0.11593238]] 0 conscientiousness

# # # check through pictures
# # plt.subplot(121), plt.title('real=%s'%(img_name)), plt.axis('off')
# # plt.imshow(img)
# # # plt.imshow(img[0].numpy().astype("uint8")) 
# # plt.subplot(122), plt.title('predict=%s:%s'%(a_predict,a_predict_name))
# # plt.plot(a_predict_array[0])    # imshow in rgb format
# # plt.show()
# # plt.savefig('predictions.png')

# val_loss, val_acc = model.evaluate(val_ds)
# print('*'*9)
# print("Validation accuracy is:\n")
# print(val_acc)

# # display 5 error predictions???
# import matplotlib.pyplot as plt
# err_cnt=0
# plt.figure(figsize=(20, 8))
# for _imgs, _labels in train_ds.take(-1): #逐个batch遍历子集
#   for i in range(len(_imgs)): #每个batch的图片
#     if err_cnt>=5: break
#     img_iname=_labels[i].numpy()
#     img_name=class_names[img_iname]
#     predictions = model.predict(tf.expand_dims(_imgs[i], 0)) 
#     img_iname_pred=np.argmax(predictions)
#     img_name_pred=class_names[img_iname_pred]
#     if img_iname!=img_iname_pred:
#       err_cnt+=1
#       ax = plt.subplot(2, 5, err_cnt)
#       plt.imshow(_imgs[i].numpy().astype("uint8"))
#       plt.title('%s->%s'%(img_name,img_name_pred))
#       plt.axis("off")
#       ax = plt.subplot(2, 5, 5+err_cnt)
#       plt.plot(predictions[0])
#       plt.savefig('predictions.png')
# print('*'*9)
# print('Finish prediction!!!')

# # pause
# import time
# time.sleep(10000)