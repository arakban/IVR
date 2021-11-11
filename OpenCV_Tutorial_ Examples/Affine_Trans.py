import cv2
import numpy as np

# function for saving left click event
x_i, y_i = -1, -1

def click_event(event, x, y, flags, param):
    global x_i, y_i
    if event == cv2.EVENT_LBUTTONDOWN:
        x_i, y_i = x, y


# read image
img = cv2.imread('hzinput.jpg', 1)

# select four points on the image and record them
coordinates_0 = np.zeros((2, 4))  # pixels in the original image
for i in range(0, 4):
    cv2.imshow('window', img)
    cv2.setMouseCallback('window', click_event)
    cv2.waitKey(3500)
    coordinates_0[:, i] = np.array([x_i, y_i])
    cv2.destroyAllWindows()

coordinates_1 = np.array([[1, 180, 180, 1], [1, 1, 250, 250]])  # pixels in the target image

# Stack up the data matrix
A = np.empty((8, 6))
b = np.empty((8, 1))
for i in range(0, 4):
    A[2 * i:2 * i + 2, :] = np.array(
        [[coordinates_1[0, i], coordinates_1[1, i], 0, 0, 1, 0], [0, 0, coordinates_1[0, i], coordinates_1[1, i], 0, 1]]
    )
    b[2 * i:2 * i + 2, 0] = coordinates_0[:, i]

# Solve for the transformation between source and target
t = np.dot(np.linalg.pinv(A), b)
Mest = np.array([[t[0, 0], t[1, 0], t[4, 0]], [t[2, 0], t[3, 0], t[5, 0]]])

# Apply the estimated transformation
out = np.empty((250, 180, 3), dtype=np.uint8)
cv2.imshow('window1', out)
for j in range(0, 180):
    for i in range(0, 250):
        src = np.dot(Mest, np.array([[j], [i], [1]]))
        index = src.astype(int)
        out[i, j, :] = img[index[1], index[0], :]

cv2.imshow('window2', img)  # show original image
cv2.imshow('window3', out)  # show projection
cv2.waitKey(0)
