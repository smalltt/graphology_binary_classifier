import cv2

x = cv2.imread('/home/lassena/yan_projects/data/2classes_split_balanced/eg1_eng_clean/split_1_2/testing_pre/extraversion/54_0.jpg', cv2.IMREAD_GRAYSCALE)
x = cv2.fastNlMeansDenoising(x, None, 3, 7, 21)
cv2.imshow('denosied', x)
cv2.imwrite('denosied_54_0.png', x)
cv2.waitKey(0)