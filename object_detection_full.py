import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as Img
import torchvision
from torchvision.transforms import functional as F
import torch.nn.functional as F_nn
import torch
import os
import time
import argparse
import pathlib
import custom_utils
import csv
from depth_est_func import depth_est
from ts_models.fasterrcnn_resnet50 import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the best model and trained weights
ts_model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('/home/jc-merlab/RBE549_P3_Einstein_Vision/helpers/outputs/resnet50_training_model/last_model.pth', map_location=DEVICE)
ts_model.load_state_dict(checkpoint['model_state_dict'])
ts_model.to(device).eval()

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.3

# For same annotation colors each time.
np.random.seed(42)
# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

cars_model = torch.hub.load("ultralytics/yolov5", "yolov5l")
cars_model.to(device)

def yolo_cars(img):
    results = cars_model(img)
    annotated_img = results.render()[0]
    label_details = results.pandas().xyxy[0]
    label_details['distance'] = label_details.apply(lambda row: depth_est(img, (row['xmin'], row['ymin'], row['xmax'], row['ymax']), row['name']), axis=1)
    # now `label_details` contains a new column `distance` with the distance value for each row
    # label_csv = label_details.to_csv(os.path.join(output_dir, "labels{:04d}.csv".format(frame_num)), index=False)
    return annotated_img, label_details

def resnet_ts(frame, label_details, resize=None):
    img = frame.copy()
    image = frame.copy()
    if resize is not None:
        img = cv2.resize(img, (resize, resize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    img /= 255.0
    # bring color channels to front
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    img = torch.tensor(img, dtype=torch.float).cuda()
    # add batch dimension
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        # get predictions for the current frame
        outputs = ts_model(img.to(DEVICE))

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        classes = outputs[0]['labels'].cpu().numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_num = classes[j]
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            frame = custom_utils.draw_boxes(frame, box, color, resize)
            frame = custom_utils.put_class_text(
                frame, box, class_name,
                color, resize
            )
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            confidence = confidence
            cls = class_num
            name = class_name
            distance = depth_est(frame, box, class_name)
            # add new row to `label_details`
            label_details.loc[len(label_details)] = [xmin, ymin, xmax, ymax, confidence, cls, name, distance]
    return frame, label_details


def main():
    # Set the path to the video file
    video_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/images2/2023-02-14_11-04-07-front_undistort.mp4"
    # Set the path to the directory where the output images will be saved
    output_dir = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/images/"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/images/scene1_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Loop through each frame of the video
    for frame_num in range(total_frames):
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Apply object detection and tracking
        annotated_img, label_details = yolo_cars(frame)
        final_labelled_img = resnet_ts(annotated_img, label_details)

        label_csv = label_details.to_csv(os.path.join(output_dir, "labels{:04d}.csv".format(frame_num)), index=False)
        # Save the output frame
        cv2.imwrite(output_dir + "frame{:04d}.jpg".format(frame_num), final_labelled_img)

        # Add the frame to the output video
        output_video.write(final_labelled_img)

    # Release the video capture and writer objects
    cap.release()
    output_video.release()


if __name__=='__main__':
    main()

