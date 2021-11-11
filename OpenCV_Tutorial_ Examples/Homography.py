import cv2
import numpy as np


# read image
img = cv2.imread('hzinput.jpg', 1)


coordinates_0=np.array([[367,508,253,131],[37,92,242,144]])

coordinates_1 = np.array([[1,180,180,1],[1,1,250,250]])  # pixels in the target image


# #cv2 provide function cv2.findHomography to calculate homografy
# pts_src = np.array([[367,30], [508,92], [253,242],[131,144]])
# pts_dst = np.array([[1,1],[180,1],[180,250],[1,250]])
# h, status = cv2.findHomography(pts_src, pts_dst)
# normh=np.linalg.norm(h)


# you can also claculate homography by yourself
A = np.empty((8, 8))
b = np.empty((8, 1))
for i in range(4):
    A[2 * i:2 * i + 2, :] = np.array([
        [coordinates_0[0, i],coordinates_0[1, i],1,0,0,0,-(coordinates_1[0, i]*coordinates_0[0, i]),-(coordinates_1[0, i]*coordinates_0[1, i])],
        [0,0,0,coordinates_0[0, i],coordinates_0[1, i],1,-(coordinates_1[1, i]*coordinates_0[0, i]), -(coordinates_1[1, i]*coordinates_0[1, i])]])
    b[2 * i:2 * i + 2, 0] = coordinates_1[:, i]

invA = np.linalg.inv(A)
h = invA@b

H0 = np.array([[h[0,0],h[1,0],h[2,0]],[h[3,0],h[4,0],h[5,0]],[h[6,0],h[7,0],1]])
normH = np.linalg.norm(H0)
H = H0 / normH


# print(img[1,1])

# def homo_trans(img_src,img_dir,H, dsize):
#     n_row=img_src.shape[0]
#     n_col=img_src.shape[1]
#     for i in range(n_row):
#         for j in range(n_col):
#             P2=H@(np.array([i,j,1]).T)
#             P2=P2.astype(int)
#             if P2[0]>0 and P2[1]> 0 and P2[0]< dsize[0] and P2[1]<dsize[1]:
#                 img_dir[P2[0],P2[1]]=img_src[i,j]
#     return img_dir

# out=np.zeros((180,250,3), np.uint8)
# homo_trans(img,out,H,(180,250))


out = cv2.warpPerspective(img, H, (180,250))
cv2.imshow('window2', img)  # show original image
cv2.imshow('window_out', out)  # show projection
cv2.waitKey(0)

