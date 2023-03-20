
img_rows, img_cols = 512, 512  # 128, 256

batch_size = 32  # 7, 8, 16, 25, 32, 64, 128
# 4 difference characters
num_classes = 2
# very short training time
epochs = 2  # 50, 100, 200, 500, 1000


model_json_path = 'model_files/model.json'
label_obj_path = 'model_files/labels.sav'
model_path = 'model_files/model.h5'

prediction_file_dir_path = 'predict_this_doc/'
