# function is for split a images to a specific pieces

import os.path
from PIL import Image
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import common_utils.folder as folder

def split_image(arg_folder_path, arg_target_size,arg_targert_folder):

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

if __name__=="__main__":
    folder_path = "/usr/test/data/eg1/training/conscientiousness"
    tile_size = 500
    targert_folder = "/usr/test/data/eg1/training/splitted_conscientiousness"

    folder.remove(targert_folder)
    folder.create(targert_folder)

    split_image(folder_path,tile_size,targert_folder)
