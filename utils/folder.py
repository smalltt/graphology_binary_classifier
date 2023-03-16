# functions for folder operation

import os
import shutil


def create(arg_folder_path):
    if not os.path.exists(arg_folder_path):
        os.makedirs(arg_folder_path)
        print(f"Folder '{arg_folder_path}' created successfully.")
    else:
        print(f"Folder '{arg_folder_path}' already exists.")

def remove(arg_folder_path):
    if os.path.exists(arg_folder_path):
        shutil.rmtree(arg_folder_path)
        print(f"Folder '{arg_folder_path}' removed successfully.")
    else:
        print(f"Folder '{arg_folder_path}' does not exist.")
