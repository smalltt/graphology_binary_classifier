## preprocessing image
import os
import time
import cv2
from scipy.misc import imresize   # conda install scipy==1.1.0

# resize + binarization
img_rows = 150
img_cols = 150
# img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/training/conscientiousness'  # 2classes_nosplit_balanced/eg1_eng_clean ; 2classes_split_balanced/eg1_eng/split_1_2
# pre_img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/training_pre/conscientiousness'
# img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/testing/conscientiousness'
# pre_img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/testing_pre/conscientiousness'
# img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/training/extraversion'
# pre_img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/training_pre/extraversion'
img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/testing/extraversion'
pre_img_path = '/home/lassena/yan_projects/data/2classes_nosplit_balanced/eg1_eng_clean/testing_pre/extraversion'

def preprocessing_img(x):
    
    # # 0. resize
    # x = imresize(x, (img_rows, img_cols))
    # cv2.imwrite('resized.png', x)

    # 1. manually remove unwanted data
    
    # 2. noise removal
    x = cv2.fastNlMeansDenoising(x, None, 3, 7, 21)
    # cv2.imshow('denoised', x)
    # cv2.imwrite('2denoised.png', x)
    # cv2.waitKey(0)
    
    # 3. crop

    # 4. binarization
    ## 1) basic global thresholding
    # __, x1 = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('simple', x1)
    # cv2.imwrite('4basic_global_th.png', x1)
    # cv2.waitKey(0)

    ## 2) Otsu global thresholding
    th, xb = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    # print('%'*9)
    # print('threshold', th)
    # cv2.imshow('otsu', xb)
    # cv2.imwrite('4otsu_global_th.png', xb)
    # cv2.waitKey(0)

    ## 3) Adaptive thresholding
    # x3 = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10) 
    # cv2.imshow('ada1', x3)
    # cv2.imwrite('4ada1_mean.png', x3)
    # cv2.waitKey(0)

    # x4 = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    # cv2.imshow('ada2', x4)
    # cv2.imwrite('4ada2_gaussian.png', x4)
    # cv2.waitKey(0)

    # 5. dilation
    morph_size = (2, 2)
    cpy = xb.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    x = ~cpy
    # cv2.imshow('dilation', x)
    # cv2.imwrite('5dilation.png', x)
    # cv2.waitKey(0)

    # 6. resize (good)
    # x = imresize(x, (img_rows, img_cols))
    # cv2.imwrite('6resized.png', x)
    # cv2.waitKey(0)
    return x

count = 0
for folder, subfolders, files in os.walk(img_path):
      for name in files:
        if name.endswith('.jpg'):
          x = cv2.imread(folder + '/' + name, cv2.IMREAD_GRAYSCALE)
          count = count + 1
          # cv2.imshow('original', x)
          # print("$"*9)
          # print(x.dtype)  # uint8
          # print(x.shape)  # (6600, 5100)
          start_time = time.time()
          x = preprocessing_img(x)  # preprocessing
          cv2.imwrite(pre_img_path+'/'+name, x)
          print('--- %s / %s ---' %(count,len(files)))
          print('has done %s for %s seconds' %(name,time.time() - start_time))
    
print('&'*9)
print('Finishing preprocessing!')