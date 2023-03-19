## preprocessing image
import os
import time
import cv2
# from scipy.misc import imresize   # conda install scipy==1.1.0

img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/testing/extraversion'
pre_img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/testing_pre/extraversion'

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

count = 0
for folder, subfolders, files in os.walk(img_path):
      for name in files:
        if name.endswith('.jpg'):
          x = cv2.imread(folder + '/' + name, cv2.IMREAD_GRAYSCALE)
          count = count + 1
          start_time = time.time()
          x = preprocessing_img(x)  # preprocessing
          cv2.imwrite(pre_img_path+'/'+name, x)
          print('--- %s / %s ---' %(count,len(files)))
          print('has done %s for %s seconds' %(name,time.time() - start_time))
    
print('&'*9)
print('Finishing preprocessing!')