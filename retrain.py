# This script is for model retraining

import tensorflow as tf
from tensorflow import keras
import conf
from common_utils import gpu as GPU
from common_utils import dataset as ds
from common_utils import folder
from model_utils import model_utils as utils


if __name__=="__main__":
    # filter tf message
    GPU.ignore_gpu_messages(3)

    # enable GPU
    gpu_enable = GPU.enable_gpu()
    if gpu_enable:
        print("GPU is enable")
    else:
        print("NO GPU is available")

    # create folder to store log after remove it
    folder.remove(conf.logs)
    folder.create(conf.logs)

    # load trained model
    model = tf.keras.models.load_model(conf.model_best_file_path)
    model.summary()

    # generate training and validation dataset
    train_ds, val_ds = ds.gen_train_val_ds(conf.train_ds_dir, conf.image_size, conf.val_spit)
    class_names = train_ds.class_names

    ds.visualize_sample_data(train_ds, class_names)
    ds.visualize_augmentated_data(train_ds, class_names)

    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

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
        tf.keras.callbacks.ModelCheckpoint(filepath=conf.model_best_path, save_best_only=True, save_weights_only=False, monitor='val_accuracy', mode='max', verbose=0),
        # # save the model for each epoch
        # keras.callbacks.ModelCheckpoint(conf.model_each_epoch_path),
    ]

    # train model
    model_fit_output = model.fit(
        train_ds,
        epochs=conf.epochs,
        callbacks=callback,
        validation_data=val_ds,
    )

    # save trained model
    utils.save_model(conf.model_json_path, conf.label_obj_path, model, conf.model_weights_path, conf.model_last_save_path)

    utils.plot_loss_accuracy(model_fit_output, conf.output_folder)
