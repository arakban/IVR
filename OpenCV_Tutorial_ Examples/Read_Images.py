import cv2
import numpy as np

# read image
img = cv2.imread('hzinput.jpg', 1)

print(img)

# Image dimentions
# get dimensions of image
dimensions = img.shape
type = img.dtype

# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]

print('Image Dimension    : ', dimensions)
print('Image Height       : ', height)
print('Image Width        : ', width)
print('Data Type        : ', type)

# show image
cv2.imshow('img_window1', img) 
cv2.waitKey(10000)
cv2.destroyAllWindows()

# write image to a file
cv2.imwrite('image_copy.png', img)

# split 3 channels and show them
b,g,r = cv2.split(img)
cv2.imshow('r_channel', r)
cv2.imshow('g_channel', g)
cv2.imshow('b_channel', b)

# Convert to grayscale
img_gray1 = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3
img_gray1 = np.around(img_gray1)
img_gray1 = img_gray1.astype(np.uint8)

img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow('img_window1', img_gray1)
cv2.imshow('img_window2', img_gray2)
cv2.imwrite('grey1.jpg', img_gray1)
cv2.imwrite('grey2.jpg', img_gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()