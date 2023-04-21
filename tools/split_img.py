# function is for split a images to a specific pieces

import os.path
from PIL import Image
import sys
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import common_utils.folder as folder
import common_utils.image as handwritingImg

def split_image(arg_folder_path, arg_target_size,arg_targert_folder, arg_removed_folder, arg_white_threshold):

    folder_path = arg_folder_path
    # Set the size of each tile
    tile_size = arg_target_size

    img_list = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, img_path) for img_path in img_list]

    for file_path in file_paths:
        if file_path[-3:] == "jpg":
            pass
        else:
            continue
        
        # Open the image file
        image = Image.open(file_path)

        # Get the size of the image
        width, height = image.size

        # Calculate the number of tiles in each dimension
        num_tiles_x = int(width / tile_size)
        num_tiles_y = int(height / tile_size)

        # Iterate over each tile
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Calculate the position of the tile
                left = i * tile_size
                upper = j * tile_size
                right = (i + 1) * tile_size
                lower = (j + 1) * tile_size
                
                # Crop the image to the tile
                tile = image.crop((left, upper, right, lower))

                # Save the tile with a unique filename
                parent_file_name = file_path.split('.')[0].split('/')[-1]
                filename = '/'+ parent_file_name +'_{}_{}.jpg'.format(i, j)
                filename = arg_targert_folder + filename
                tile.save(filename)
                print("-"*10)
                print(filename)

                img_white_percent = handwritingImg.white_percent(filename)

                if img_white_percent > arg_white_threshold:
                    # Delete image file
                    # os.remove(filename)
                    image_name = os.path.basename(filename)
                    print("="*8)
                    print(image_name)
                    removed_file_path = os.path.join(arg_removed_folder,image_name)
                    print(removed_file_path)
                    os.rename(filename, removed_file_path)
                    print('{0} white percent is deleted, with white percent {1}'.format(filename, img_white_percent))

if __name__=="__main__":
    folder_path = "/usr/test/data/eg1/training/preprocessed_conscientiousness"
    targert_folder = "/usr/test/data/eg1/training/splitted_conscientiousness"
    removed_folder = "/usr/test/data/eg1/training/removed_conscientiousness"

    # folder_path = "/usr/test/data/eg1/training/preprocessed_extraversion"
    # targert_folder = "/usr/test/data/eg1/training/splitted_extraversion"
    # removed_folder = "/usr/test/data/eg1/training/removed_extraversion"

    tile_size = 800
    white_threshold = 97

    folder.remove(targert_folder)
    folder.create(targert_folder)

    folder.remove(removed_folder)
    folder.create(removed_folder)

    split_image(folder_path,tile_size,targert_folder,removed_folder, white_threshold)
