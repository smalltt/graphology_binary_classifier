# This script is for model training

from utils import gpu as GPU
from utils import dataset as ds
from models import model1
import conf
import tensorflow as tf
from tensorflow import keras
from models import model_utils as utils

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

    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    '''
    import warnings
    # Filter out the warning message
    warnings.filterwarnings("ignore", message="Using a while_loop for converting")
    '''

    ds.visualize_augmentated_data(train_ds, class_names)

    # build model
    model = model1.make_model(arg_input_shape=conf.input_shape, arg_num_classes=len(class_names))

    # plot model network
    keras.utils.plot_model(
        model, to_file='model_files/model.png', show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False
        )

    # model compile
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint("model_files/save_at/save_at_{epoch}.keras"),
    ]

    # train model
    model_fit_output = model.fit(
        train_ds,
        epochs=conf.epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    # save trained model
    utils.save_model(conf.model_json_path, conf.label_obj_path, model, conf.model_weights_path, conf.model_last_save_path)
    
    utils.plot_loss_accuracy(model_fit_output, conf.epochs)
