# configuration variable

# data source path
# train_ds_dir = "/usr/test/data/eg1_mnist/training"
# test_ds_dir = "/usr/test/data/eg1_mnist/testing"

# train_ds_dir = "/usr/test/data/eg1/training"
# test_ds_dir = "/usr/test/data/eg1/testing"

train_ds_dir = "/usr/test/data/dataset_gra/split_1_2/training"  # no_split, split_1_2, split_3_4, split_6_8, split_9_12
test_ds_dir = "/usr/test/data/dataset_gra/split_1_2/testing"

val_spit=0.2

# check if the folder exists
import os
from model_utils import model_self_def, model_convnexttiny

output_folder = 'output'
# model_json_path = os.path.join(output_folder, 'model.json')
# label_obj_path = os.path.join(output_folder, 'labels.h5')
# model_each_epoch_path = os.path.join(output_folder, 'each_epoch', 'each_epoch_model_at_{epoch}.h5')
model_each_epoch_path = os.path.join(output_folder, 'each_epoch_model', 'each_epoch_model_at_{epoch}')
# model_weights_path = os.path.join(output_folder, 'last_epoch_model_weights.h5')
# model_last_epoch_path  = os.path.join(output_folder, 'last_epoch_model_at_{epoch}.h5')
model_last_epoch_path  = os.path.join(output_folder, 'last_epoch_model_at_{epoch}')
# model_best_epoch_path = os.path.join(output_folder, 'best_model', 'best_epoch_model_at_{epoch}.h5')
model_best_epoch_path = os.path.join(output_folder, 'best_epoch_model', 'best_epoch_model_at_{epoch}')

# specify a model for evaluation and inference
# model_evaluation_inference_path = os.path.join(output_folder, 'best_model', 'best_model_at_1.h5')
model_evaluation_inference_path = os.path.join(output_folder, 'best_model', 'best_model_at_1')

# logs
logs = "logs"

# hyperparameters for modeling
image_size = 150
input_shape = (image_size, image_size, 3)
input_im_size = (image_size, image_size)
batch_size = 32
model_choose = model_convnexttiny  # model_self_def, model_convnexttiny
epoch_choose = model_best_epoch_path  # model_best_epoch_path, model_last_epoch_path
epochs = 2
