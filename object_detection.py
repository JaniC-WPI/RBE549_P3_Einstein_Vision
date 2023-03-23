import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Set the path to the video file
# video_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4"

# # Set the path to the directory where the output images will be saved
# output_dir = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/images/"

# # Create the output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Set the paths to the YOLOv5 configuration and weights files
# config_file = "/home/jc-merlab/yolov5/models/yolov5s.yaml"
# weights_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/yolov5s.pt"

# # Load the YOLOv5 network
# net = cv2.dnn_DetectionModel(config_file, weights_file)
# net.setInputSize(640, 640)
# net.setInputScale(1.0 / 255)

# # Open the video file
# cap = cv2.VideoCapture(video_file)

# # Loop through the frames in the video
# frame_num = 1
# while True:
#     # Read the next frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect objects in the frame using YOLOv5
#     classes, scores, boxes = net.detect(frame, confThreshold=0.5, nmsThreshold=0.5)

#     # Draw bounding boxes around the detected objects
#     colors = np.random.uniform(0, 255, size=(len(boxes), 3))
#     for i, box in enumerate(boxes):
#         left, top, width, height = box
#         color = colors[i]
#         cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
#         label = "{}: {:.2f}".format(classes[i][0], scores[i])
#         cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Save the frame as an image file
#     output_file = os.path.join(output_dir, "frame{:04d}.jpg".format(frame_num))
#     cv2.imwrite(output_file, frame)

#     # Increment the frame counter
#     frame_num += 1

# # Release the video file and cleanup
# cap.release()
# cv2.destroyAllWindows()

img = cv2.imread('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/images/frame0519.jpg')

plt.imshow(img)
plt.show()

