# import necessary packages
# from turtle import color
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np



def gen_train_val_ds(arg_ds_dir, arg_image_size, arg_val_spit=0.2):
    ## Split samples as training and validation datasets
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=arg_ds_dir,
        image_size=(arg_image_size, arg_image_size),
        shuffle=True,
        seed=123,
        validation_split=arg_val_spit,
        subset="both",
        )

    return train_ds, val_ds

def gen_test_ds(arg_ds_dir, arg_image_size, arg_batch_size):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=arg_ds_dir,  
        batch_size=arg_batch_size,
        image_size=(arg_image_size, arg_image_size),
        seed=123, # seed for what?
        )

    return test_ds

def pre_data(arg_train_ds_dir,arg_test_ds_dir,arg_image_size,arg_val_spit,arg_batch_size):
    # generate training and validation dataset
    train_ds, val_ds = gen_train_val_ds(arg_train_ds_dir, arg_image_size, arg_val_spit)

    # generate testing dataset
    test_ds = gen_test_ds(arg_test_ds_dir, arg_image_size, arg_batch_size)

    '''
    # get lable of test dataset
    # take(?) should be used properly
    test_lable_ls = []
    # i = 0
    for y_test_image, y_test_label in test_ds:
        for item in y_test_label.numpy():
            test_lable_ls.append(item)
    # print(len(test_lable_ls))
    # print(test_lable_ls)

    # get class name list
    class_names = train_ds.class_names
    num_classes = len(class_names)
    # print(class_names)  # ['conscientiousness', 'extraversion']
    '''

    return train_ds, val_ds, test_ds

def enable_gpu():
    # enable GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    except:
        return False

def visualize_sample_data(arg_train_ds, arg_class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in arg_train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title( arg_class_names[labels[i]])  ## 0: conscientiousness; 1: extraversion     
            plt.axis("off")
            plt.savefig('first_9_sample_images.png')

def fetch_base_model(arg_model_name, arg_weights, arg_classes_nb, arg_input_shape):
    base_model = keras.applications.ConvNeXtTiny(
    model_name=arg_model_name,
    include_top=False,
    include_preprocessing=True,
    weights=arg_weights,
    input_tensor=None,
    input_shape=arg_input_shape,
    pooling=None,
    classes=arg_classes_nb,
    classifier_activation="softmax",
    )

    return base_model

def save_model(arg_model,arg_model_weights_path,arg_model_default_save_path):
    # serialize model to JSON
    model_json = arg_model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)

    # pickle label encoder obj
    with open(label_obj_path, 'wb') as lb_obj:
        pickle.dump(lb, lb_obj)

    # serialize weights to HDF5
    arg_model.save_weights(arg_model_weights_path)
    arg_model.save(arg_model_weights_path)
    arg_model.save(model_default_save_path)
    
# plot loss and accuracy
def plot_loss_accuracy(arg_model_fit_output, arg_nb_epochs):
    plt.style.use("ggplot")
    plt.figure()
    H = arg_model_fit_output
    N = arg_nb_epochs
    plt.plot(np.arange(0, N), H.history["loss"], color='lightgrey', label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], color='blue', label="val_loss")
    plt.plot(np.arange(0, N), H.history["binary_accuracy"], color='red', label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_binary_accuracy"], color='green', label="val_acc")
    plt.legend()
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    plt.savefig('model_files/training_loss_acc.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # enable GPU
    gpu_enable = enable_gpu()
    if gpu_enable:
        print("GPU is enable")
    else:
        print("NO GPU is available")

    # data source path
    train_ds_dir = "/usr/test/eg1_mnist/training"
    test_ds_dir = "/usr/test/eg1_mnist/testing"
    val_spit=0.2

    # check if the folder exists
    model_json_path = 'model_files/model.json'
    label_obj_path = 'model_files/labels.sav'
    # model_path = 'model_files/model.h5'
    model_weights_path = 'model_files/model_weights.h5'
    model_default_save_path  = 'model_files/tf_model.save'

    IMAGE_SIZE = 150
    INPUT_SHAPE = (150, 150, 3)
    size = (150, 150)
    BATCH_SIZE = 32
    epochs = 2

    # parameters of model
    model_name = "convnext_tiny"
    model_weights = "imagenet"
    nb_classes = 2 # len(class_names)

    lb = LabelEncoder()
    # print('lb = ',lb)

    # prepare data
    train_ds, val_ds, test_ds = pre_data(train_ds_dir,test_ds_dir,IMAGE_SIZE,val_spit,BATCH_SIZE)
    
    class_names = train_ds.class_names
    visualize_sample_data(train_ds, class_names)

    test_lable_ls = []
    for y_test_image, y_test_label in test_ds:
        for item in y_test_label.numpy():
            test_lable_ls.append(item)

    ## Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),  
        layers.RandomRotation(0.1), 
        # layers.RandomTranslation(height_factor=(-0.2, 0.3)),  # has problem???
        layers.RandomContrast(factor=[0.1, 0.9]),
        layers.Rescaling(scale=1 / 255)  # [0, 1]
        # layers.Rescaling(scale=1 / 127.5, offset=-1),  #[-1, 1]   
    ])

    base_model = fetch_base_model(model_name, model_weights, nb_classes, INPUT_SHAPE)
    ## Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = data_augmentation(inputs)

    # ConvNeXt models expect their inputs to be float or uint8 tensors of pixels with values in the [0-255] range. 
    # # the rescaling layer outputs: `(inputs * scale) + offset`
    # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)  # [-1, 1]
    # x = scale_layer(x)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1, name="precisions")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    
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
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    callbacks=[
        #给tensorboard仪表盘提供数据, how to use???
        tf.keras.callbacks.TensorBoard(log_dir='./tf_cnn_board', histogram_freq=1),
        #执行过程中动态调整LearningRate。 本例是初始0.001，每10轮lr减半???
        # tf.keras.callbacks.LearningRateScheduler(lambda epoch:0.001/np.power(2,int(epoch/10))),
        # tf.keras.callbacks.LearningRateScheduler(lambda epoch:0.001/np.power(2,int(epoch/10))),  ## training samples different from my own dataset
        # #提前终止条件。本例是验证集的预测精度超过5轮都没有提升了，就终止???
        # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.0001,patience=5,restore_best_weights=True),
        # #每训练一轮，就把当前的模型及参数保存一次（通过save_freq控制每轮保存次数）
        # tf.keras.callbacks.ModelCheckpoint(filepath="tf_model_epoch-{epoch:04d}", verbose=1, save_weights_only=False, save_freq=50)
    ]

    # # Define a callback to save the best model
    # checkpoint_callback =[
    #     tf.keras.callbacks.TensorBoard(log_dir='./tf_cnn_board', histogram_freq=1),
    #     tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', save_best_only=True, save_weights_only=False, monitor='val_accuracy', mode='max', verbose=1)
    # ]

    # train model
    model_fit_output = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )
    # print('Model training completed successfully')

    # save trained model
    save_model(model,model_weights_path,model_default_save_path)
    
    plot_loss_accuracy(model_fit_output, epochs)


    # load trained model
    loaded_model = tf.keras.models.load_model(model_default_save_path)
    loaded_model.summary()
    
    # test trained model
    score = loaded_model.evaluate(test_ds, verbose=1)  
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # # predict test dataset
    # predict_results = model.predict(test_ds)
    # print(len(predict_results))

    # predict test dataset
    predict_results = loaded_model.predict(test_ds)
    print(predict_results>0.5)

    predict_ls = predict_results>0.5

    result_ls=[]
    for result in predict_ls:
        if result[0]:
            result_ls.append(1)
        else:
            result_ls.append(0)

    
    print(result_ls)
    print(test_lable_ls)

    # print(test_lable_ls)

    img = keras.preprocessing.image.load_img(
        "eg1_mnist/testing/0/3.png", target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = loaded_model.predict(img_array)
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% 0 and {100 * score:.2f}% 1.")
