# This script is for model training

import os
from common_utils import gpu as GPU
from common_utils import dataset as ds
from common_utils import folder
from model_utils import model1
import conf
import tensorflow as tf
from tensorflow import keras
from model_utils import model_utils as utils


if __name__ == "__main__":

    # filter tf message
    GPU.ignore_gpu_messages(3)

    # enable GPU
    gpu_enable = GPU.enable_gpu()
    if gpu_enable:
        print("GPU is enable")
    else:
        print("NO GPU is available")

    # create folder to store output after remove it
    folder.remove(conf.output_folder)
    folder.create(conf.output_folder)

    # create folder to store log after remove it
    folder.remove(conf.logs)
    folder.create(conf.logs)

    # generate training and validation dataset
    train_ds, val_ds = ds.gen_train_val_ds(conf.train_ds_dir, conf.image_size, conf.val_spit)
    class_names = train_ds.class_names

    ds.visualize_sample_data(train_ds, class_names)
    ds.visualize_augmentated_data(train_ds, class_names)

    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    '''
    import warnings
    # Filter out the warning message
    warnings.filterwarnings("ignore", message="Using a while_loop for converting")
    '''

    # build model
    model = model1.make_model(arg_input_shape=conf.input_shape, arg_num_classes=len(class_names))

    # plot model network
    keras.utils.plot_model(
        model, to_file= os.path.join(conf.output_folder, 'model.png'), show_shapes=False,
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
        # optimizer=keras.optimizers.Adam(),
        # loss=keras.losses.BinaryCrossentropy(),
        # metrics=[keras.metrics.BinaryAccuracy()],
    )

    callback =[
        # save logs
        tf.keras.callbacks.TensorBoard(log_dir=conf.logs, histogram_freq=1),
        # save the best model
        tf.keras.callbacks.ModelCheckpoint(filepath=conf.model_best_path, save_best_only=True, save_weights_only=False, monitor='val_accuracy', mode='max', verbose=1),
        # # save the model for each epoch
        # keras.callbacks.ModelCheckpoint(conf.model_each_epoch_path),
        # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=2, mode='auto', restore_best_weights=True),
    ]

    # train model
    model_fit_output = model.fit(
        train_ds,
        epochs=conf.epochs,
        callbacks=callback,
        validation_data=val_ds,
    )

    # save trained model
    utils.save_model(model, conf.model_last_epoch)

    utils.plot_loss_accuracy(model_fit_output,conf.output_folder)
