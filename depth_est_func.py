import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the monocular depth estimation model (replace with your own model)
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
# model.to(device)

def depth_est(image, box, class_name):
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])
    class_img = image[ymin:ymax, xmin:xmax]

    # Convert the image to grayscale and resize to match the depth model input size
    class_img = cv2.cvtColor(class_img, cv2.COLOR_BGR2GRAY)
    class_img = cv2.merge([class_img, class_img, class_img])
    class_img = cv2.resize(class_img, (384, 384))
    # Convert the image to a PyTorch tensor and normalize
    class_tensor = F.to_tensor(class_img).unsqueeze(0)
    class_tensor = F.normalize(class_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Run the depth estimation model on the image
    with torch.no_grad():
        depth_tensor = model(class_tensor)
    # Convert the depth tensor to a numpy array and resize to match the original bounding box size
    depth_array = depth_tensor.squeeze().cpu().numpy()
    depth_array = cv2.resize(depth_array, (xmax - xmin, ymax - ymin))
    # Scale the depth values to the original image size
    scale_factor_x = image.shape[1] / 384
    scale_factor_y = image.shape[0] / 384
    depth_array = cv2.resize(depth_array, (0, 0), fx=scale_factor_x, fy=scale_factor_y)
    # Calculate the distance to the car as the median depth value within the bounding box
    distance_m = np.median(depth_array)

    return distance_m
