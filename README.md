# How to run tensorboard
- 1. open image
- $ sudo docker run --gpus all -it -v "$PWD":/usr/test --rm -p 6006:6006 tensorflow/graphology-tensorflow:2.11.0-gpu
- 2. change dir
- $ cd /usr/test
- 3. background running tensorboard
- $ nohup tensorboard --logdir=/usr/test/logs/ --bind_all &
- 4. open local browser and input:
- http://127.0.0.1:6006

## train.py
- to use keras.utils.plot_model(), install:
- $ pip install pydot
- $ pip install pydotplus
- $ sudo apt-get install graphviz
- $ sudo apt-get update

# Under the folder "tools":
## MNIST_generator.py
- function: generate MNIST dataset locally

## split_img.sh
- function: split multiple images by rows and columns.
- usage: split-image ./must_be_img.jpg 2 1 --output-dir ./auto_generate_target_dir  # 2 rows 1 column
- 1. install split-image(refer to:https://pypi.org/project/split-image/)
- $ conda create -n env_split_img python==3.6.9
- $ conda activate env_split_img
- $ pip install split-image
- 2. change the permissions on .sh file to make it executable
- $ chmod u+x split_img.sh
- 3. open terminal under the image files and run .sh file
- $ /sh_file_dir/split_img.sh

## preprocess_img.py
- function: preprocess image by denoise, binarization, dilation, (resize).
- install opencv with python==3.6.9
- $ pip install opencv-python

## copy_img.py
- function: copy images with the names in the folder of 'name_path' from the folder of 'img_path' into the folder of 'copy_path'.
- install opencv with python==3.6.9
