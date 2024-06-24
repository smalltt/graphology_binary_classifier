# This script is for noise removal.
# refer to: https://www.kaggle.com/code/chanduanilkumar/adding-and-removing-image-noise-in-python/notebook
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_path = '../data/temp/7_7.jpg'
# imread an image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print(img.shape)
cv2.imwrite('orig_gray.png', img)

# 1. Gaussian noise
gauss_noise=np.zeros(img.shape, dtype=np.uint8)
cv2.randn(gauss_noise,128,20)   # with a mean of 128 and a sigma of 20
gauss_noise=(gauss_noise*0.5).astype(np.uint8)
gn_img=cv2.add(img, gauss_noise)
cv2.imwrite('gauss_noise.png', gauss_noise)
cv2.imwrite('gn_img.png', gn_img)

# 2. uniform noise (not often encountered in real-world imaging systems)
uni_noise=np.zeros(img.shape,dtype=np.uint8)
cv2.randu(uni_noise,0,255)
uni_noise=(uni_noise*0.5).astype(np.uint8) 
un_img=cv2.add(img,uni_noise)
cv2.imwrite('uni_noise.png', uni_noise)
cv2.imwrite('un_img.png', un_img)

# 3. salt-and-pepper noise(impulse noise)
imp_noise=np.zeros(img.shape,dtype=np.uint8)
cv2.randu(imp_noise,0,255)
imp_noise=cv2.threshold(imp_noise,245,255,cv2.THRESH_BINARY)[1]
in_img=cv2.add(img, imp_noise)
cv2.imwrite('imp_noise.png', imp_noise)
cv2.imwrite('in_img.png', in_img)

# noise removal
# 1. inbuilt fastNlMeansDenoising(Slightly effective against Gaussian Noise)
# x = cv2.fastNlMeansDenoising(img, None, 23, 7, 21)
denoised_gn_fast = cv2.fastNlMeansDenoising(gn_img, None, 10, 10)
denoised_un_fast = cv2.fastNlMeansDenoising(un_img, None, 10, 10)
denoised_in_fast = cv2.fastNlMeansDenoising(in_img, None, 10, 10)
cv2.imwrite('denoised_gn_fast.png', denoised_gn_fast)
cv2.imwrite('denoised_un_fast.png', denoised_un_fast)
cv2.imwrite('denoised_in_fast.png', denoised_in_fast)

# 2. median blur(Slightly effective against Gaussian Noise, considerably effective against Impulse Noise)
denoised_gn_median=cv2.medianBlur(gn_img,3)
denoised_un_median=cv2.medianBlur(un_img,3)
denoised_in_median=cv2.medianBlur(in_img,3)
cv2.imwrite('denoised_gn_median.png', denoised_gn_median)
cv2.imwrite('denoised_un_median.png', denoised_un_median)
cv2.imwrite('denoised_in_median.png', denoised_in_median)

# 3. Gaussian blur(Slightly Effective against Gaussian Blur)
denoised_gn_gaussian=cv2.GaussianBlur(gn_img,(3,3),0)
denoised_un_gaussian=cv2.GaussianBlur(un_img,(3,3),0)
denoised_in_gaussian=cv2.GaussianBlur(in_img,(3,3),0)
cv2.imwrite('denoised_gn_gaussian.png', denoised_gn_gaussian)
cv2.imwrite('denoised_un_gaussian.png', denoised_un_gaussian)
cv2.imwrite('denoised_in_gaussian.png', denoised_in_gaussian)

# display
fig=plt.figure(dpi=300)
# 1
fig.add_subplot(5,3,1)
plt.imshow(gauss_noise,cmap='gray')
plt.axis("off")
plt.title("gn")

fig.add_subplot(5,3,2)
plt.imshow(uni_noise,cmap='gray')
plt.axis("off")
plt.title("un")

fig.add_subplot(5,3,3)
plt.imshow(imp_noise,cmap='gray')
plt.axis("off")
plt.title("in")

# 2
fig.add_subplot(5,3,4)
plt.imshow(gn_img,cmap='gray')
plt.axis("off")
plt.title("gn_img")

fig.add_subplot(5,3,5)
plt.imshow(un_img,cmap='gray')
plt.axis("off")
plt.title("un_img")

fig.add_subplot(5,3,6)
plt.imshow(in_img,cmap='gray')
plt.axis("off")
plt.title("in_img")

# 3
fig.add_subplot(5,3,7)
plt.imshow(denoised_gn_fast,cmap='gray')
plt.axis("off")
plt.title("denoised_gn_fast")

fig.add_subplot(5,3,8)
plt.imshow(denoised_un_fast,cmap='gray')
plt.axis("off")
plt.title("denoised_un_fast")

fig.add_subplot(5,3,9)
plt.imshow(denoised_in_fast,cmap='gray')
plt.axis("off")
plt.title("denoised_in_fast")

# 4
fig.add_subplot(5,3,10)
plt.imshow(denoised_gn_median,cmap='gray')
plt.axis("off")
plt.title("denoised_gn_median")

fig.add_subplot(5,3,11)
plt.imshow(denoised_un_median,cmap='gray')
plt.axis("off")
plt.title("denoised_un_median")

fig.add_subplot(5,3,12)
plt.imshow(denoised_in_median,cmap='gray')
plt.axis("off")
plt.title("denoised_in_median")

# 5
fig.add_subplot(5,3,13)
plt.imshow(denoised_gn_gaussian,cmap='gray')
plt.axis("off")
plt.title("denoised_gn_gaussian")

fig.add_subplot(5,3,14)
plt.imshow(denoised_un_gaussian,cmap='gray')
plt.axis("off")
plt.title("denoised_un_gaussian")

fig.add_subplot(5,3,15)
plt.imshow(denoised_in_gaussian,cmap='gray')
plt.axis("off")
plt.title("denoised_in_gaussian")

plt.savefig('denoised_imgs.png')