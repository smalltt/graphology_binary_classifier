For keras.utils.plot_model()

pip install pydot
pip install pydotplus
sudo apt-get install graphviz

sudo apt-get update


Steps:

1.
sudo docker run --gpus all -it -v "$PWD":/usr/test --rm tensorflow/graphology-tensorflow:2.11.0-gpu
2.
cd /usr/test
