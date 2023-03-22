# import cv2
# import numpy as np
# import os

# # Define the dimensions of the checkerboard
# checkerboard_size = (9, 6)

# # Define the object points of the checkerboard corners
# objp = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
# objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

# # Define the arrays to store the object points and image points
# obj_points = []   # 3D points in real world space
# img_points = []   # 2D points in image plane

# # Loop through each checkerboard image
# image_dir = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Calib/right/"
# image_files = os.listdir(image_dir)
# for image_file in image_files:
#     # Read the image
#     image_path = os.path.join(image_dir, image_file)
#     img = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Find the corners of the checkerboard in the image
#     ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

#     # If the corners are found, add them to the object and image points arrays
#     if ret == True:
#         obj_points.append(objp)
#         img_points.append(corners)

# # Get the camera calibration parameters
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# # Get the rotation and translation vectors for the camera
# rvec, tvec = rvecs[0], tvecs[0]

# # Convert the rotation vector to a rotation matrix
# R, _ = cv2.Rodrigues(rvec)

# # Invert the rotation matrix to get the camera orientation
# R_inv = np.linalg.inv(R)

# # Get the camera position in world coordinates
# pos = -np.dot(R_inv, tvec)

# # Print the camera position
# print("Camera position:", pos)

import cv2
import numpy as np
import os
import glob

# set the path to the directory containing the checkerboard images
path = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Calib/left/"

# set the size of the checkerboard (number of internal corners)
nx = 9
ny = 6

# create the object points for the corners of the checkerboard
objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# create empty lists for the object points and image points
objpoints = []
imgpoints = []

# loop over the images in the directory
for filename in glob.glob(os.path.join(path, '*.jpg')):
    # read the image
    img = cv2.imread(filename)
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find the corners of the checkerboard
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # if the corners are found, add the object points and image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# get the camera matrix and distortion coefficients using the image points and object points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print the camera matrix and distortion coefficients
print("Camera Matrix:")
print(mtx)
print("Distortion Coefficients:")
print(dist)

# get the rotation matrix and translation vector for the camera
rmat, _ = cv2.Rodrigues(rvecs[0])
tvec = tvecs[0]

# get the camera position in world frame
cam_pos = -rmat.T @ tvec

# print the camera position in world frame
print("Camera Position in World Frame:")
print(cam_pos)