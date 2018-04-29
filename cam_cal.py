import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# Create object coordinates
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Store object and image coordinates in arrays
objpoints = []
imgpoints = []

# Import list of calibration image file names using glob
images = glob.glob('../CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')

# Search for chessboard corners in each image
for idx, fname in enumerate(images):   
    nx = 9 # No. of column corners
    ny = 6 # No. of row corners
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If corners are found, input coordinates into arrays
    if ret == True:
    	imgpoints.append(corners)
    	objpoints.append(objp) 
    	cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    	write_name = 'corners_found'+str(idx)+'.jpg'
    	cv2.imwrite(write_name, img)

img = cv2.imread(images[0])
img_size = (img.shape[1], img.shape[0]) # Shape of all calibration images

# Calibrate the camera. Return camera matrix and distortion matrix
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, img_size, None, None)

# Save as pickle file
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))

# example_image = 'corners_found13.jpg'
# img = cv2.imread(example_image)
# undist_example = cv2.undistort(img, mtx, dist, None, mtx)
# cv2.imwrite('undistorted_board_example.jpg', undist_example)