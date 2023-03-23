import cv2
import numpy as np
import os
import torch
from PIL import Image 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the path to the video file
video_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene2/Undist/2023-03-03_10-31-11-front_undistort.mp4"

# Set the path to the directory where the output images will be saved
output_dir = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene2/images/"

model = torch.hub.load("ultralytics/yolov5", "yolov5l")
model.to(device)

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Apply Gaussian blurring to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # # Apply Canny edge detection to detect edges
    edges = cv2.Canny(blur, 50, 150)

    cv2.imshow("edges", edges)
    cv2.waitKey(2)

    # # Define the ROI mask
    height, width = edges.shape
    print(height, width)
    mask = np.zeros_like(edges)
    roi_corners = np.array([[(0, 900), (width, 900), (600, 450)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)

    # cv2.imshow("masks", mask)
    # cv2.waitKey(2)

    # # Apply the ROI mask to the edges
    masked_edges = cv2.bitwise_and(edges, mask)

    # cv2.imshow("masks", masked_edges)
    # cv2.waitKey(2)

    # # Define the Hough line detection parameters
    rho = 2
    theta = np.pi/180
    threshold = 100
    min_line_length = 40
    max_line_gap = 5

    # # Apply Hough line detection to the ROI
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # line_image = np.zeros_like(frame)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line.reshape(4)
    #         cv2.line(line_image, (x1,y1), (x2, y2), (255, 0, 0), 10)

    # img = cv2.addWeighted(frame, 0.7, line_image, 1.0, 1)

    # cv2.imshow("line image", line_image)
    # cv2.waitKey(2)    

    # # Separate the lines into left and right lanes
    if lines is not None:
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            params = np.polyfit((x1,x2), (y1,y2), 1)
            slope = params[0]
            intercept = params[1]
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))


        if not right_lines:
            right_lines = [(0.7804878048780496, -72.70731707317177), (0.779069767441859, -62.151162790695295), (0.8395061728395045, -103.20987654320734)]

        if not left_lines:
            left_lines = [(-0.7252252252252256, 904.2342342342347), (-0.6818181818181818, 888.8181818181821), (-0.6039603960396039, 851.069306930693), (-0.6888888888888915, 890.7111111111122), (-0.7543859649122828, 908.3859649122811), (-0.7288135593220344, 906.3389830508477)]

        print("right lines", right_lines)
        print("left lines", left_lines)
        left_slope, left_intercept = np.average(left_lines, axis=0)
        right_slope, right_intercept =  np.average(right_lines, axis=0)

        print(right_slope, right_intercept)

        ly1 = frame.shape[0]
        # ly2 = int(ly1*(3/5))
        ly2 = int(ly1*0.7)
        lx1 = int((ly1-left_intercept)/left_slope)
        lx2 = int((ly2-left_intercept)/left_slope)

        ry1 = frame.shape[0]
        # ry2 = int(ry1*(3/5))
        ry2 = int(ry1*0.7)
        rx1 = int((ry1-right_intercept)/right_slope)
        rx2 = int((ry2-right_intercept)/right_slope)

        print(rx1, ry1, rx2, ry2)
        print(rx2 > -900)
        print(rx2<800)

        if rx1 > 1280:
            rx1 = 1280
        elif rx1 < -1280:
            rx1 = -1280
        if (rx2 < 900) or (rx2 > 950):         
            rx2 = 900
        elif (rx2 < -950):         
            rx2 = -900

        if lx1 > 1280:
            lx1 = 1280
        elif lx1 < -1280:
            lx1 = -1280
        if (lx2 < 450) or (lx2 > 950):
            lx2 = 340
        elif (lx2 < -950):
            lx2 = -340


        # printrx1, ry1, rx2, ry2)
        left_line =  np.array([lx1, ly1, lx2, ly2])
        right_line = np.array([rx1, ry1, rx2, ry2])

        if len(right_lines) == 0:
           right_line = np.array([3829,  960, -164,  576])
        else:
            right_line = np.array([rx1, ry1, rx2, ry2])

        print(right_line)
        print(left_line)

        averaged_lines = np.array([left_line, right_line])

        line_image = np.zeros_like(frame)
        if averaged_lines is not None:
            for line in averaged_lines:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1,y1), (x2, y2), (255, 0, 0), 10)

        img = cv2.addWeighted(frame, 0.7, line_image, 1.0, 1)

    else:
        img = frame

    results = model(img)
    # Render the annotated image with bounding boxes and labels
    annotated_image = results.render()[0]
    label_details = results.pandas().xyxy[0]
    if lines is not None:
        # Add right lane label
        label_details = label_details.append({
            'xmin': rx1,
            'xmax': rx2,
            'ymin': ry1,
            'ymax': ry2,
            'confidence': 0.7,
            'class': -1,
            'name': 'right lane'
        }, ignore_index=True)

        # Add left lane label
        label_details = label_details.append({
            'xmin': lx1,
            'xmax': lx2,
            'ymin': ly1,
            'ymax': ly2,
            'confidence': 0.7,
            'class': -2,
            'name': 'left lane'
        }, ignore_index=True)
    # print(type(label_details))
    label_csv = label_details.to_csv(os.path.join(output_dir, "labels{:04d}.csv".format(frame_num)), index=False)
    print(type(label_csv))
    print(label_details)
    # Save the annotated image to a file
    output_file = os.path.join(output_dir, "frame{:04d}.jpg".format(frame_num))
    cv2.imwrite(output_file, annotated_image)

    # Increment the frame counter
    frame_num += 1

    print(frame_num)

    # if frame_num > 100:
    #     break

# Release the video file and cleanup
cap.release()
cv2.destroyAllWindows()