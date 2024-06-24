# This script is for model evaluation

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import conf
from common_utils import dataset as ds
import json

tf.get_logger().setLevel('ERROR')  # clear warnings

if __name__=="__main__":
    # load trained model
    loaded_model = tf.keras.models.load_model(conf.epoch_chosen)
    loaded_model.summary()

    # Generate the testing dataset
    test_ds = ds.gen_test_ds(conf.test_ds_dir, conf.image_size_width, conf.image_size_height, conf.batch_size)

    # run model evaluation
    score = loaded_model.evaluate(test_ds, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # get label from test_ds
    y_test = []
    for y_test_image, y_test_label in test_ds.take(-1):
        y_test = y_test + y_test_label.numpy().tolist()
    
    # evaluate the network
    out = loaded_model.predict(test_ds)
    y_pred = out.argmax(axis=1)
    print('='*9)
    print('out.argmax(axis=1): ', out.argmax(axis=1))  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    print("[INFO] evaluating network...")
    print(classification_report(y_test, y_pred))
    
    # Create a string containing the results
    results_text = f"Test loss: {score[0]}\nTest accuracy: {score[1]}\nPredictions: {y_pred.tolist()}\nClassification report:\n{classification_report(y_test, y_pred)}"

    # Specify the file path (change this to your desired target folder)
    results_path = r"./output/test_result.txt"

    # Write the results to the text file
    with open(results_path, "w") as file:
        file.write(results_text)

    print(f"Results saved to {results_path}")    
        
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(len(cm)), yticklabels=range(len(cm)))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the confusion matrix as a JPG image
    plt.savefig('./output/confusion_matrix.jpg', format='jpg')
    plt.close()

    print("Confusion matrix saved as confusion_matrix.jpg")