# This script is for model training

from utils import gpu as GPU
from utils import dataset as ds
import conf
import tensorflow as tf

if __name__ == "__main__":

    # filter tf message
    GPU.ignore_gpu_messages(3)

    # enable GPU
    gpu_enable = GPU.enable_gpu()
    if gpu_enable:
        print("GPU is enable")
    else:
        print("NO GPU is available")

    # generate training and validation dataset
    train_ds, val_ds = ds.gen_train_val_ds(conf.train_ds_dir, conf.image_size, conf.val_spit)
    # generate testing dataset
    test_ds = ds.gen_test_ds(conf.test_ds_dir, conf.image_size, conf.batch_size)

    class_names = train_ds.class_names

    ds.visualize_sample_data(train_ds, class_names)

    '''
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    '''

    '''
    import warnings
    # Filter out the warning message
    warnings.filterwarnings("ignore", message="Using a while_loop for converting")
    '''

    ds.visualize_augmentated_data(train_ds, class_names)