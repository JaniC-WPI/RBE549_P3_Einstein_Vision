import cv2
import torch
'''
To predict the depth information of a scene from a video of a running car in Python, 
you can use a pre-trained depth estimation model and a video processing library such as OpenCV. Here's an example code:
'''




# Load pre-trained depth estimation model
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Open video file
cap = cv2.VideoCapture('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4')

# Loop through frames and predict depth map
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame to fit model input size
    frame = cv2.resize(frame, (384, 384))
    
    # Convert image to torch tensor and normalize
    input_tensor = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0) / 255.0
    
    # Move tensor to device
    input_tensor = input_tensor.to(device)
    
    # Predict depth map using model
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Convert depth map to numpy array
    depth_map = prediction.squeeze().cpu().numpy()
    
    # Normalize depth map for display
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Display depth map
    cv2.imshow('Depth Map', depth_map)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''This code uses the MiDaS (Mixed Depth Estimation and Stereo) model for depth estimation, which is a state-of-the-art model for 
monocular depth estimation. The model is loaded using the torch.hub.load function from the PyTorch Hub, and is moved to the GPU if available.

The code then opens the video file using OpenCV, and loops through the frames of the video. For each frame, the code resizes the frame 
to the input size of the model (384x384), converts the frame to a torch tensor, and normalizes the tensor. The tensor is then moved to the 
GPU and passed through the model to predict the depth map. The depth map is converted to a numpy array and normalized for display. 
The depth map is displayed using OpenCV's imshow function.'''