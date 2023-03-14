For keras.utils.plot_model()

pip install pydot
pip install pydotplus
sudo apt-get install graphviz

sudo apt-get update


Steps:

1.
sudo docker run --gpus all -it -v "$PWD":/usr/test --rm -p 6006:6006 tensorflow/graphology-tensorflow:2.11.0-gpu

2.
cd /usr/test

3.
nohup tensorboard --logdir=/usr/test/logs/ --bind_all &

4.
http://<backend-server-ip>:6006

http://127.0.0.1:6006


run
python tools/MNIST_generator.py
to generate MNIST dataset locally
