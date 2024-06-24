# for error

# import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report

tf.get_logger().setLevel('ERROR')  # clear warnings

BATCH_SIZE = 2048  # 32(*), 64, 128, 256, 512, 700, 1024, 2048
# split9_12: 2042; split6_8: 1044; split3_4: 328; split1_2: 58; no_split: 28
IMAGE_SIZE = 150

test_ds = image_dataset_from_directory(
    directory="/usr/test/data/2classes_split_balanced/eg1/split_9_12/testing",  # 2classes_split_balanced/eg1/split_1_2/testing
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    seed=123,
)   
# Found 28 files belonging to 2 classes.

for y_test_image, y_test_label in test_ds.take(1):
    print('='*9)
    print('y_test_image shape: ', y_test_image.numpy().shape)  # y_test_image shape:  (28, 150, 150, 3)
    print('y_test_label: ', y_test_label.numpy().shape)   # y_test_label:  (28,)

print('^'*9)
print('y_test_label: ', y_test_label)  # y_test_label:  tf.Tensor([0 0 0 0 1 1 1 1 1 1 0 1 1 1 0 1 0 0 1 0 0 1 0 1 0 1 0 0], shape=(28,), dtype=int32)
print('y_test_label.numpy(): ', y_test_label.numpy())  # y_test_label.numpy():  [0 0 0 0 1 1 1 1 1 1 0 1 1 1 0 1 0 0 1 0 0 1 0 1 0 1 0 0]
y_test = y_test_label.numpy()
print('len(y_test): ', len(y_test))  # len(y_test):  28

# load model
model = load_model('tf_model.save')

# evaluate the network

BATCH_SIZE = 32  # 32(*), 64, 128, 256, 512, 700, 1024, 2048
test_ds = image_dataset_from_directory(
    directory="/usr/test/data/2classes_split_balanced/eg1/split_9_12/testing",  # 2classes_nosplit_balanced/eg1/testing
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    seed=123,
)
out = model.predict(test_ds)
print('!'*9)
print('out.argmax(axis=1): ', out.argmax(axis=1))  # out.argmax(axis=1):  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
print('len(out.argmax(axis=1)): ', len(out.argmax(axis=1)))  # len(out.argmax(axis=1)):  28
print("[INFO] evaluating network...")
if len(y_test)<len(out.argmax(axis=1)):
	print(classification_report(y_test, out.argmax(axis=1)[:len(y_test)]))
else:
	print(classification_report(y_test, out.argmax(axis=1)))

