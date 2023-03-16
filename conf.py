# configuration variable

# data source path
# train_ds_dir = "/usr/test/data/eg1_mnist/training"
# test_ds_dir = "/usr/test/data/eg1_mnist/testing"
train_ds_dir = "/usr/test/data/eg1/training"
test_ds_dir = "/usr/test/data/eg1/testing"
val_spit=0.2

# check if the folder exists
import os
output_folder = 'output'
model_json_path = os.path.join(output_folder, 'model.json')
label_obj_path = os.path.join(output_folder, 'labels.h5')
model_each_epoch_path = os.path.join(output_folder, 'each_epoch', 'model_at_{epoch}.h5')
model_weights_path = os.path.join(output_folder, 'last_epoch_model_weights.h5')
model_last_save_path  = os.path.join(output_folder, 'last_epoch_tf_model.h5')
model_best_path = os.path.join(output_folder, 'best_model', 'best_model_at_{epoch}.h5')
model_best_file_path = os.path.join(output_folder, 'best_model', 'best_model_at_2.h5')

# logs
logs = "logs"

# hyperparameters for modeling
image_size = 150
input_shape = (image_size, image_size, 3)
input_im_size = (image_size, image_size)
batch_size = 16
epochs = 2
