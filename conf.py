# configuration variable

# data source path
train_ds_dir = "/usr/test/data/conscientiousness-extraversion/train"  # no_split, split_1_2, split_3_4, split_6_8, split_9_12
test_ds_dir = "/usr/test/data/conscientiousness-extraversion/test"

val_split=0.2

# check if the folder exists
import os
from model_utils import model_self_def, model_convnexttiny, model_xception, model_vgg16, model_resnet50v2, model_inceptionv3, model_mobilenetv2, model_densenet121, model_nasnetmobile  # , model_efficientnetb0

output_folder = 'output'
# model_json_path = os.path.join(output_folder, 'model.json')
# label_obj_path = os.path.join(output_folder, 'labels.h5')
model_each_epoch_path = os.path.join(output_folder, 'each_epoch_model', 'each_epoch_model_at_{epoch}')
# model_weights_path = os.path.join(output_folder, 'last_epoch_model_weights.h5')
model_last_epoch_path  = os.path.join(output_folder, 'last_epoch_model_at_{epoch}')
model_best_epoch_path = os.path.join(output_folder, 'best_epoch_model', 'best_epoch_model_at_{epoch}')

# specify a model for evaluation and inference
model_evaluation_inference_path = os.path.join(output_folder, 'best_model', 'best_model_at_1')

# logs
logs = "logs"

# hyperparameters for modeling
image_size_width = 850
image_size_height = 825

scale_ratio = 0.5
image_size_width = int(image_size_width * scale_ratio)
image_size_height = int(image_size_height * scale_ratio)

# image_size = 150
input_shape = (image_size_height, image_size_width, 3)
input_im_size = (image_size_width, image_size_height)
batch_size = 16
model_chosen = model_convnexttiny  # model_self_def, model_convnexttiny, model_xception, model_vgg16, model_resnet50v2, model_inceptionv3, model_mobilenetv2, model_densenet121, model_nasnetmobile, model_efficientnetb0
epoch_chosen = model_best_epoch_path  # model_best_epoch_path, model_last_epoch_path
# pred_img_path = 'data/eg1/testing/conscientiousness/1.jpg'  # conscientiousness/7.jpg, extraversion/5.jpg
pred_img_path = 'data/splitted_Bennie_Peleman/out3_0_1.jpg'
epochs = 100
