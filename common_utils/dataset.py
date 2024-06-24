# Functions for dataset processing--bk

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import conf


#     return train_ds, val_ds

def gen_train_val_ds(arg_ds_dir, arg_image_size_wight, arg_image_size_height, arg_val_split=0.2, num_classes=None):
    ## Split samples as training and validation datasets
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=arg_ds_dir,
        image_size=(arg_image_size_height,arg_image_size_wight),
        shuffle=True,
        seed=123,
        validation_split=arg_val_split,
        subset="both",
        )
    # Extract class names before mapping
    class_names = train_ds.class_names
    
    return train_ds, val_ds, class_names

# def gen_test_ds(arg_ds_dir, arg_image_size, arg_batch_size):
#     test_ds = tf.keras.utils.image_dataset_from_directory(
#         directory=arg_ds_dir,  
#         batch_size=arg_batch_size,
#         image_size=(arg_image_size, arg_image_size),
#         seed=123,
#         )

#     return test_ds

def gen_test_ds(arg_ds_dir, arg_image_size_wight, arg_image_size_height, arg_batch_size, num_classes=None):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=arg_ds_dir,  
        batch_size=arg_batch_size,
        image_size=(arg_image_size_height, arg_image_size_wight),
        seed=123,
        )

    return test_ds

def visualize_sample_data(arg_train_ds, arg_class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in arg_train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title( arg_class_names[labels[i]]) 
            plt.axis("off")
            plt.savefig(conf.output_folder+'/Sample_images.png')

def visualize_augmentated_data(arg_train_ds, arg_class_names):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomContrast(factor=[0.1, 0.9]),
            # layers.Rescaling(1.0 / 255),
        ]
    )

    plt.figure(figsize=(10, 10))
    for images, labels in arg_train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[i].numpy().astype("uint8"))
            plt.title( arg_class_names[labels[i]])     
            plt.axis("off")
            plt.savefig(conf.output_folder+'/Augmentated_images.png')
