import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the image and the CSV file
img = cv2.imread('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene2/images/frame0523.jpg')
df = pd.read_csv('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene2/images/labels0523.csv')

# Load the monocular depth estimation model (replace with your own model)
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
# model.to(device)

resolution = 1920

# Iterate through the rows of the CSV file
for index, row in df.iterrows():
    if row['name'] == 'car':
        # Extract the bounding box coordinates
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])

        # Crop the image to the bounding box
        car_img = img[ymin:ymax, xmin:xmax]

        # Convert the image to grayscale and resize to match the depth model input size
        car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
        car_img = cv2.merge([car_img, car_img, car_img])
        car_img = cv2.resize(car_img, (384, 384))

        # Convert the image to a PyTorch tensor and normalize
        car_tensor = F.to_tensor(car_img).unsqueeze(0)
        car_tensor = F.normalize(car_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Run the depth estimation model on the image
        with torch.no_grad():
            depth_tensor = model(car_tensor)

        # Convert the depth tensor to a numpy array and resize to match the original bounding box size
        depth_array = depth_tensor.squeeze().cpu().numpy()
        depth_array = cv2.resize(depth_array, (xmax - xmin, ymax - ymin))

        distance_m = np.median(depth_array)

        print(distance_m)

        print(img.shape)
        # Scale the depth values to the original image size
        scale_factor_x = img.shape[1] / 384
        scale_factor_y = img.shape[0] / 384
        depth_array = cv2.resize(depth_array, (0, 0), fx=scale_factor_x, fy=scale_factor_y)

        # Calculate the distance to the car as the median depth value within the bounding box
        distance_m = np.median(depth_array)

        print(distance_m)

        # Estimate the size of the car in meters (replace with your own estimate)
        car_size_m = 4

        # Calculate the size of the car in pixels
        car_size_px = (car_size_m / distance_m) * resolution

        # Draw a line from the bottom center of the image to the bottom center of the car
        start_pt = (int((img.shape[1]) / 2), img.shape[0])
        end_pt = (int((xmin + xmax) / 2), int(ymax - car_size_px))        
        color = (0, 0, 255)  # Red color for the line
        thickness = 2
        cv2.line(img, start_pt, end_pt, color, thickness)

        # Print the distance to the car in pixels
        print(f"Distance to car {index}: {car_size_px} pixels")
        print("actual distance in pixels", start_pt, end_pt)

    if row['name'] == 'traffic light':
        # Extract the bounding box coordinates
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])

        # Crop the image to the bounding box
        class_img = img[ymin:ymax, xmin:xmax]

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

        print(img.shape)
        # Scale the depth values to the original image size
        scale_factor_x = img.shape[1] / 384
        scale_factor_y = img.shape[0] / 384
        depth_array = cv2.resize(depth_array, (0, 0), fx=scale_factor_x, fy=scale_factor_y)

        # Calculate the distance to the car as the median depth value within the bounding box
        distance_m = np.median(depth_array)

        print(distance_m)

        # Estimate the size of the car in meters (replace with your own estimate)
        class_size_m = 4

        # Calculate the size of the car in pixels
        class_size_px = (class_size_m / distance_m) * resolution

        # Draw a line from the bottom center of the image to the bottom center of the car
        start_pt = (int((img.shape[1]) / 2), img.shape[0])
        end_pt = (int((xmin + xmax) / 2), int(ymax - car_size_px))        
        color = (0, 255, 0)  # Red color for the line
        thickness = 2
        cv2.line(img, start_pt, end_pt, color, thickness)

        # Print the distance to the car in pixels
        print(f"Distance to trafic lights {index}: {car_size_px} pixels")
        print("actual distance in pixels", start_pt, end_pt)


# Display the image with the line
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()