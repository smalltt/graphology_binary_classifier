## preprocessing image
import os
import time
import cv2
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import common_utils.folder as folder

def preprocessing_img(x):
    
    # noise removal
    x = cv2.fastNlMeansDenoising(x, None, 3, 7, 21)
    
    # binarization: Otsu global thresholding
    th, xb = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
 
    # dilation
    morph_size = (2, 2)
    cpy = xb.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    x = ~cpy

    return x

def start(arg_img_path,arg_pre_img_path):
  count = 0
  for folder, subfolders, files in os.walk(arg_img_path):
        for name in files:
          if name.endswith('.jpg') or name.endswith('.bmp'):
            x = cv2.imread(folder + '/' + name, cv2.IMREAD_GRAYSCALE)
            count = count + 1
            start_time = time.time()
            x = preprocessing_img(x)  # preprocessing
            cv2.imwrite(arg_pre_img_path+'/'+name, x)
            print('--- %s / %s ---' %(count,len(files)))
            print('has done %s for %s seconds' %(name,time.time() - start_time))

if __name__=="__main__":
  folder_path = "/usr/test/data/Bennie_Peleman"
  targert_folder = "/usr/test/data/preprocess_Bennie_Peleman"

  # folder_path = "/usr/test/data/eg1/training/extraversion"
  # targert_folder = "/usr/test/data/eg1/training/preprocessed_extraversion"

  folder.remove(targert_folder)
  folder.create(targert_folder)

  start(folder_path,targert_folder)

  print('&'*9)
  print('Finishing preprocessing!')