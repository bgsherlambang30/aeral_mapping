
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# Import required modules
import cv2
import numpy as np


calib_data = np.load("dataOut/calib.npz")
mat = calib_data["matrix"]
dist = calib_data["distortion"]

img = cv2.imread('data/calibTest/1637691998.098.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mat, dist, (w, h), 1, (w, h))
# undistort
dst = cv2.undistort(img, mat, dist, None, newcameramtx)

cv2.imwrite('dataOut/calibresult.jpg', dst)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('dataOut/calibresult_cropped.jpg', dst)
