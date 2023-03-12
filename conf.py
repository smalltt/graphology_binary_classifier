# configuration variable

# data source path
train_ds_dir = "/usr/test/eg1_mnist/training"
test_ds_dir = "/usr/test/eg1_mnist/testing"
# train_ds_dir = "/usr/test/eg1/training"
# test_ds_dir = "/usr/test/eg1/testing"
val_spit=0.2

# check if the folder exists
model_json_path = 'model_files/model.json'
label_obj_path = 'model_files/labels.save'
# model_path = 'model_files/model.h5'
model_weights_path = 'model_files/model_weights.h5'
model_default_save_path  = 'model_files/tf_model.save'

image_size = 150
input_shape = (150, 150, 3)
# size = (150, 150)
batch_size = 32
epochs = 2

