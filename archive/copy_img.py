## copy images with the names in the folder of 'name_path' from the folder of 'img_path' into the folder of 'copy_path'.
import os
import time
import cv2

name_path = '/home/lassena/Desktop/yan_projects/data/2classes_split_balanced/eg1_eng/split_9_12/training/extraversion'
copy_path = '/home/lassena/yan_projects/data/2classes_split_balanced/eg1_eng/split_9_12/training_pre/extraversion'
img_path = '/home/lassena/yan_projects/data/2classes_split_balanced/eg1_eng_pre/split_9_12/extraversion'

count = 0
for folder, subfolders, files in os.walk(name_path):
      for name in files:
        if name.endswith('.jpg'):
          count = count + 1
          start_time = time.time()
          x = cv2.imread(img_path + '/' + name)
          cpy = x.copy()
          cv2.imwrite(copy_path + '/'+name, cpy)
          print('--- %s / %s ---' %(count,len(files)))
          print('has done %s for %s seconds' %(name,time.time() - start_time))
    
print('&'*9)
print('Finishing copy!')
