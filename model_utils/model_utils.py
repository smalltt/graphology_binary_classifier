# Function for modeling

import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os

def save_model(arg_model,arg_model_last_model_path):
# def save_model(arg_model_json_path,arg_label_obj_path,arg_model,arg_model_weights_path,arg_model_last_model_path):
    # # serialize model to JSON
    # model_json = arg_model.to_json()
    # with open(arg_model_json_path, "w") as json_file:
    #     json_file.write(model_json)

    # lb = LabelEncoder()
    # # pickle label encoder obj
    # with open(arg_label_obj_path, 'wb') as lb_obj:
    #     pickle.dump(lb, lb_obj)

    # # serialize weights to HDF5
    # arg_model.save_weights(arg_model_weights_path)
    # arg_model.save(arg_model_weights_path)
    arg_model.save(arg_model_last_model_path)

def plot_loss_accuracy(arg_model_fit_output, arg_output_folder):
    plt.style.use("ggplot")
    plt.figure()
    H = arg_model_fit_output
    N = len(H.history["loss"])
    plt.plot(np.arange(0, N), H.history["loss"], color='lightgrey', label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], color='blue', label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], color='red', label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], color='green', label="val_acc")
    # plt.plot(np.arange(0, N), H.history["binary_accuracy"], color='red', label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_binary_accuracy"], color='green', label="val_acc")
    plt.legend()
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    plt.savefig(os.path.join(arg_output_folder, 'training_loss_acc.png'), bbox_inches='tight')
    plt.show()
