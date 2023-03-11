# This script is for model training

from utils import gpu as GPU

if __name__ == "__main__":

    # GPU.ignore_gpu_messages(2)
    GPU.ignore_gpu_messages(3)
    # enable GPU
    gpu_enable = GPU.enable_gpu()
    if gpu_enable:
        print("GPU is enable")
    else:
        print("NO GPU is available")
