# function for image processing

import cv2
import numpy as np
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
        print(hist)
    else:
        print('Image histogram is above threshold.')
        print(hist)

def white_percent(arg_image_path):
    img = cv2.imread(arg_image_path, cv2.IMREAD_GRAYSCALE)

    lower_gray = 128
    upper_gray = 255

    mask = np.logical_and(img >= lower_gray, img <= upper_gray)

    white_area_percent = (np.count_nonzero(mask) / (img.shape[0] * img.shape[1])) * 100

    return white_area_percent

# not work
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


if __name__=="__main__":
    img_path = "/usr/test/data/eg1/training/splitted_conscientiousness/14_0_1.jpg"
    # hist = hist_value(img_path)
    # print(hist)

    white_per = white_percent(img_path)
    print(white_per)