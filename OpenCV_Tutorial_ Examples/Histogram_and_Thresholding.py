import cv2
import numpy as np
from matplotlib import pyplot as plt


# read image
img = cv2.imread('test.png', 1)
# img = cv2.imread('test1.jpg', 1)
cv2.imshow('img_window1', img)

# Image thresholding (set value of pixels that are between 120 to 255 to 255)
ret,thresh = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
cv2.imshow('img_window2', thresh)

# set a mask
mask = np.zeros(img.shape[:2],np.uint8)
mask[500:700,600:900] = 255

# calculate histogram of image
hist = cv2.calcHist([img],[2],None,[256],[0,256])
hist1 = cv2.calcHist([img],[2],mask,[256],[0,256])
plt.plot(hist)
plt.show()
plt.plot(hist1)
plt.show()
#plt.hist(img.ravel(),256,[0,256])



plt.show()
plt.xlabel("Pixel value")
plt.ylabel("Number of pixels")