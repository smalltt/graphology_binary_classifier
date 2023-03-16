# functions for folder operation

import os


def folder_create(arg_folder_path):
    if not os.path.exists(arg_folder_path):
        os.makedirs(arg_folder_path)
        print(f"Folder '{arg_folder_path}' created successfully.")
    else:
        print(f"Folder '{arg_folder_path}' already exists.")

def folder_remove(arg_folder_path):
    if os.path.exists(arg_folder_path):
        os.rmdir(arg_folder_path)
        print(f"Folder '{arg_folder_path}' removed successfully.")
    else:
        print(f"Folder '{arg_folder_path}' does not exist.")
