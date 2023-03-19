# function for image processing

import cv2
import os

def remove_bs_hist_from_file(arg_img_path,arg_hist_threshold):
    # Load image
    img = cv2.imread(arg_img_path)

    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Set threshold value
    threshold = arg_hist_threshold

    # Compare histogram to threshold
    if hist.sum() < threshold:
        # Delete image file
        os.remove(arg_img_path)
        print('Image deleted.')
    else:
        print('Image histogram is above threshold.')

def remove_bs_hist_from_memory(arg_img_in_memory,arg_hist_threshold):
    print('here')
    # Load image
    # img = cv2.imread(arg_img_in_memory)

    # Calculate histogram
    hist = cv2.calcHist([arg_img_in_memory], [0], None, [256], [0, 256])

    # Set threshold value
    threshold = arg_hist_threshold

    # Compare histogram to threshold
    if hist.sum() < threshold:
        # Delete image file
        # os.remove(arg_img_path)
        print('Image deleted.')
        return False
    else:
        print('Image histogram is above threshold.')
        return True
