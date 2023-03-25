# import cv2
# import numpy as np
# import tensorflow as tf

# # Load the pre-trained model
# model = tf.keras.models.load_model('/home/jc-merlab/Autopilot-TensorFlow/save/model.ckpt')

# # Load the test image
# image = cv2.imread('/home/jc-merlab/YOLOv5-Model-with-Lane-Detection/data/images/example_01.jpg')

# # Resize the image to the input size of the model
# input_size = (512, 256)
# image = cv2.resize(image, input_size)

# # Normalize the image
# image = image.astype(np.float32) / 255.0

# # Make a prediction using the model
# prediction = model.predict(np.expand_dims(image, axis=0))[0]

# # Threshold the prediction to create a binary image
# threshold = 0.5
# binary_image = (prediction > threshold).astype(np.uint8) * 255

# # Apply morphological operations to smooth and fill gaps in the lane markings
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# # Find the lane lines using a Hough transform
# rho = 1
# theta = np.pi / 180
# threshold = 50
# min_line_length = 100
# max_line_gap = 50
# lines = cv2.HoughLinesP(binary_image, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

# # Draw the lane lines on the original image
# line_color = (0, 0, 255)
# line_thickness = 5
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(image, (x1, y1), (x2, y2), line_color, line_thickness)

# # Display the result
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg") 

# img = results.render()[0]

# from scipy.misc import imresize
import numpy as np
# from  moviepy.editor import VideoFileClip
import cv2
from tensorflow import keras
model =  keras.models.load_model('/home/jc-merlab/RBE549_P3_Einstein_Vision/test/model.h5')