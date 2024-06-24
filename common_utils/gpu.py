# Functions defined for GPU

import os
import tensorflow as tf

def ignore_gpu_messages(arg_level):
    '''
    The TF_CPP_MIN_LOG_LEVEL environment variable can be set to one of four values:

    0 (default): Log all messages (INFO, WARNING, and ERROR).
    1: Filter out INFO messages.
    2: Filter out INFO and WARNING messages.
    3: Filter out all messages except for ERROR messages.
    '''

    msg_level = str(arg_level)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = msg_level

def enable_gpu():
    # enable GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    except:
        return False
