#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:06:24 2021
@author: pedrofRodenas
"""
import os

import pandas as pd
from PIL import Image
import tensorflow as tf

def make_dirs(path):
# def make_dirs(path_list):
    # for path in path_list:
    if not os.path.exists(path):
        os.makedirs(path)

def make_containing_dirs(path_list):
    
    for path in path_list:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

class SaverMNIST():
    def __init__(self, image_train_path, image_test_path, csv_train_path, 
                 csv_test_path):
        
        self._image_format = '.png'
        
        self.store_image_paths = [image_train_path, image_test_path]
        self.store_csv_paths = [csv_train_path, csv_test_path]
        
        for i in range(10):
            make_dirs(os.path.join(image_train_path, str(i)))

        # import sys
        # sys.exit(0)

        make_containing_dirs(self.store_csv_paths)
             
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist
        self.data = mnist.load_data()
        
    def run(self):
        
        for collection, store_image_path, store_csv_path in zip(self.data, 
                                                                self.store_image_paths,
                                                                self.store_csv_paths):
            
            labels_list = []
            paths_list = []
            
            for index, (image, label) in enumerate(zip(collection[0], 
                                                       collection[1])):
                im = Image.fromarray(image)
                width, height = im.size
                image_name = str(index) + self._image_format
                
                # Build save path
                save_path = os.path.join(store_image_path, str(label), image_name)
                print(save_path)
                im.save(save_path)
                
                labels_list.append(label)
                paths_list.append(save_path)
                
            df = pd.DataFrame({'image_paths':paths_list, 'labels': labels_list})
            
            df.to_csv(store_csv_path)
            
if __name__ == '__main__':

    
    mnist_saver = SaverMNIST(image_train_path='dataset/images',
                             image_test_path='dataset/images',
                             csv_train_path='dataset/train.csv',
                             csv_test_path='dataset/test.csv')
    
    # Write files into disk
    mnist_saver.run()