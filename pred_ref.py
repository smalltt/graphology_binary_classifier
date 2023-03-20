# Score the model
import os
# import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import load_img, to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

from tensorflow.keras import layers

# # model
# inputs = keras.Input(shape=(784,), name="digits")
# x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
# x = layers.Dense(64, activation="relu", name="dense_2")(x)
# outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

# model = keras.Model(inputs=inputs, outputs=outputs)


# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# print('*'*9)
# print('x_test', x_test)
# print(x_test.shape)  # (10000, 28, 28)
# print(x_test.dtype)   # uint8

# x_train = x_train.reshape(60000, 784).astype("float32") / 255
# x_test = x_test.reshape(10000, 784).astype("float32") / 255

# y_train = y_train.astype("float32")
# y_test = y_test.astype("float32")

# print('+'*9)
# print('x_test', x_test)
# print(x_test.shape)  # (10000, 784)
# print(x_test.dtype)  # float32

# # Reserve 10,000 samples for validation
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]

# # specify the training configuration (optimizer, loss, metrics):
# model.compile(
#     optimizer=keras.optimizers.RMSprop(),  # Optimizer
#     # Loss function to minimize
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     # List of metrics to monitor
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )

# # train the model by slicing the data into "batches" of size batch_size, and repeatedly iterating over the entire dataset for a given number of epochs.
# print("Fit model on training data")
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     epochs=2,
#     # We pass some validation for
#     # monitoring validation loss and metrics
#     # at the end of each epoch
#     validation_data=(x_val, y_val),
# )


# # Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=32)
# print("test loss, test acc:", results)  # test loss, test acc: [0.14496901631355286, 0.9559999704360962]

# # Generate predictions (probabilities -- the output of the last layer)
# # on new data using `predict`
# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test[:3])
# print("predictions shape:", predictions.shape)  # predictions shape: (3, 10)
# print('predictions', predictions)  # predictions [[9.65892166e-09 1.20138912e-08 9.50631329e-06 6.15425743e-05
#  #  1.13853271e-09 4.16657606e-08 2.28222866e-12 9.99925017e-01
#  #  1.31212188e-08 3.87217096e-06]
#  # [1.43708485e-05 2.46249081e-04 9.94762719e-01 4.88193659e-03
#  #  1.42866163e-09 7.77163805e-05 2.46795435e-06 1.95811808e-07
#  #  1.43620391e-05 2.78969932e-08]
#  # [7.01311073e-05 9.65028763e-01 8.56919773e-03 2.44292011e-03
#  #  9.57503682e-04 3.44708242e-04 1.24713383e-03 1.73061211e-02
#  #  3.63015570e-03 4.03306942e-04]]



test_dir = "/usr/test/data/2classes_nosplit_balanced/eg1/testing"
X_test, y_test = [], []

for root, directories, files in os.walk(test_dir):
   for directory in directories:
       image_files = os.listdir(os.path.join(root, directory))
       for image_file in image_files: 
            image_path = os.path.join(root, directory, image_file)
            image_data = load_img(image_path)
            X_test.append(image_data)
            y_test.append(directory)

X_test1 = np.asarray(X_test)
y_test1 = np.asarray(y_test)

print('*'*9)
print('X_test', X_test)
print('X_test shape', np.shape(X_test))  
print('X_test type', np.dtype(X_test))
print(y_test.shape)  
print(y_test.dtype)

X_test1 = np.array(X_test1, dtype="float32") / 255.0
# X_test1 = X_test1.astype('float32')
# X_test1 /= 255

print('+'*9)
print('X_test1', X_test1)
print(X_test1.shape)  
print(X_test1.dtype)

lb = LabelEncoder()
y_test1 = lb.fit_transform(y_test1)
y_test1 = to_categorical(y_test1)

print('&'*9)
print('y_test1', y_test1)
print(y_test1.shape)  
print(y_test1.dtype)

# pause
import time
time.sleep(10000)


# import glob
 
# image_list = []
# label_list = []
 
# path = 'dataset/'

# for folder in os.listdir(path):
#     for img in glob.glob(os.path.join(path, folder, '*.jpg')):
#         image_list.append(img)
#         label_list.append(folder)

# Load the model
model = load_model('tf_model.save')

# Score the model
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_test1, y_test1, batch_size=32)
print("Test loss, Test acc:", results)



preds = model.predict(X_test)
score = accuracy_score(y_test, preds)
print('Model Accuracy: %.2f' % score)


score = model.score(X, y)
print("Model accuracy: {}".format(score))



test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/usr/test/data/2classes_nosplit_balanced/eg1/testing",  # 2classes_nosplit_balanced/eg1, 2classes_split_balanced/eg2/split_6_8
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=False,
    seed=1777,
    # validation_split=0.2,
    # subset="both",
)

class_names = train_ds.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)        
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.title(class_names[int(labels[0])])  
        plt.axis("off")
        plt.savefig('augmented_image0.png')
print('*'*9)
print('Finish visualizing augmented images!!!')


# Score the model
score = model.score(X, y)
print("Model accuracy: {}".format(score))

from sklearn.metrics import classification_report

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(classification_report(y_true, y_pred))

