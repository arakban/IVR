import cv2
import numpy as np

# read image
img = cv2.imread('rgb_src.jpg',1)
cv2.imshow('img', img)
cv2.waitKey(1000)
height = img.shape[0]
width = img.shape[1]


# RGB normalization
img_rgb = img
for i in range(height):
    for j in range(width):
        r = int(img[i,j,2])
        g = int(img[i,j,1])
        b = int(img[i,j,0])
        sumrgb=r+b+g+0.00001
        r = np.around(r/sumrgb*255)
        g = np.around(g/sumrgb*255)
        b = np.around(b/sumrgb*255)
           
        img_rgb[i,j,2] = r.astype(np.uint8)
        img_rgb[i,j,0] = b.astype(np.uint8)
        img_rgb[i,j,1] = g.astype(np.uint8)

# b,g,r = cv2.split(img)
# img_rgb = cv2.merge((blueValue,greenValue,redValue))

cv2.imwrite("rgb_norm.jpg",img_rgb)
cv2.imshow('img with rgb normalization', img_rgb)
cv2.waitKey(0)
