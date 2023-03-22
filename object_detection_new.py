import cv2
import numpy as np
import os
import torch
from PIL import Image 
import torchvision.transforms as transforms

# Set the path to the video file
video_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene2/Undist/2023-03-03_10-31-11-front_undistort.mp4"

# Set the path to the directory where the output images will be saved
output_dir = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/images/"

model = torch.hub.load("ultralytics/yolov5", "yolov5l")

# model = torch.load("/home/jc-merlab/YOLOv5-Model-with-Lane-Detection/yolov5s.pt")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Open the video file
cap = cv2.VideoCapture(video_file)

# Loop through the frames in the video
frame_num = 1
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Apply Gaussian blurring to reduce noise
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # # Apply Canny edge detection to detect edges
    # edges = cv2.Canny(blur, 50, 150)

    # cv2.imshow("edges", edges)

    # # Define the ROI mask
    # height, width = edges.shape
    # mask = np.zeros_like(edges)
    # roi_corners = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    # cv2.fillPoly(mask, roi_corners, 255)

    # # Apply the ROI mask to the edges
    # masked_edges = cv2.bitwise_and(edges, mask)

    # # Define the Hough line detection parameters
    # rho = 2
    # theta = np.pi/180
    # threshold = 50
    # min_line_length = 100
    # max_line_gap = 50

    # # Apply Hough line detection to the ROI
    # lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # # Separate the lines into left and right lanes
    # left_lines = []
    # right_lines = []
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     slope = (y2 - y1) / (x2 - x1)
    #     if slope < 0 and x1 < width/2 and x2 < width/2:
    #         left_lines.append(line)
    #     elif slope > 0 and x1 > width/2 and x2 > width/2:
    #         right_lines.append(line)

    # # Calculate the average slope and intercept for each set of lines
    # left_slope = np.mean([((line[0][3] - line[0][1]) / (line[0][2] - line[0][0])) for line in left_lines])
    # left_intercept = np.mean([line[0][1] - left_slope * line[0][0] for line in left_lines])
    # right_slope = np.mean([((line[0][3] - line[0][1]) / (line[0][2] - line[0][0])) for line in right_lines])
    # right_intercept = np.mean([line[0][1] - right_slope * line[0][0] for line in right_lines])

    # # Define the extrapolation points for the left and right lanes
    # y1 = height
    # y2 = height/2
    # left_x1 = int((y1 - left_intercept) / left_slope)
    # left_x2 = int((y2 - left_intercept) / left_slope)
    # right_x1 = int((y1 - right_intercept) / right_slope)
    # right_x2 = int((y2 - right_intercept) / right_slope)

    # # Draw the lanes on the image
    # lane_image = np.zeros_like(frame)
    # cv2.line(lane_image, (left_x1, y1), (left_x2, y2), (0, 255, 0), 10)
    # cv2.line(lane_image, (right_x1, y1), (right_x2, y2), (0, 255, 0), 10)
    # result = cv2.addWeighted(frame, 0.8, lane_image, 1, 0)

    # Detect objects
    # print(model.summary())
    results = model(frame)

    # Render the annotated image with bounding boxes and labels
    annotated_image = results.render()[0]

    # Save the annotated image to a file
    output_file = os.path.join(output_dir, "frame{:04d}.jpg".format(frame_num))
    cv2.imwrite(output_file, annotated_image)

    # Increment the frame counter
    frame_num += 1

# Release the video file and cleanup
cap.release()
cv2.destroyAllWindows()