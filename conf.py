# configuration variable

# data source path
# train_ds_dir = "/usr/test/data/eg1_mnist/training"
# test_ds_dir = "/usr/test/data/eg1_mnist/testing"
train_ds_dir = "/usr/test/data/eg1/training"
test_ds_dir = "/usr/test/data/eg1/testing"
val_spit=0.2

# check if the folder exists
model_json_path = 'model_files/model.json'
label_obj_path = 'model_files/labels.save'
model_each_epoch_path = "model_files/each_epoch/model_at_{epoch}.save"
model_weights_path = 'model_files/last_epoch_model_weights.h5'
model_last_save_path  = 'model_files/last_epoch_tf_model.save'
model_best_path = "model_files/best_model/best_model_at_{epoch}.save"

# logs
logs = "logs"

image_size = 2550
input_shape = (2550, 2550, 3)
input_im_size = (2550, 2550)
batch_size = 2
epochs = 3000
