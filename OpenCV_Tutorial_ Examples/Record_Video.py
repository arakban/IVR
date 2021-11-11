import cv2
import numpy as np

# Connect to camera
cap = cv2.VideoCapture(-1)

while True:
    # Record a frame
    ret, frame = cap.read()
    cv2.imshow('window', frame)
    Exit = cv2.waitKey(200)

    # if s key pressed stop
    if Exit == ord('s'):
        cv2.destroyAllWindows()
        break


# # Filtering blue color in the image
# # define range of blue color in HSV
# lower_blue = np.array([100, 0, 0])
# upper_blue = np.array([255, 100, 100])
#
# # Threshold the HSV image to get only blue colors
# mask = cv2.inRange(frame, lower_blue, upper_blue)
#
# # Bitwise-AND mask and original image
# res = cv2.bitwise_and(frame, frame, mask=mask)
#
# cv2.imshow('frame', frame)
# cv2.imshow('mask', mask)
# cv2.imshow('res', res)
#
# while True:
#     Exit = cv2.waitKey(0)
#     # if s key pressed stop
#     if Exit == ord('s'):
#         cv2.destroyAllWindows()
#         break

cap.release()
cv2.destroyAllWindows()
