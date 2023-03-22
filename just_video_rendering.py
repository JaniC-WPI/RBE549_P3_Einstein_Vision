import cv2
import numpy as np

# Load the video files
front_video = cv2.VideoCapture("/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4")
back_video = cv2.VideoCapture("/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-back_undistort.mp4")
left_video = cv2.VideoCapture("/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-left_repeater_undistort.mp4")
right_video = cv2.VideoCapture("/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-right_repeater_undistort.mp4")

# Get video properties (e.g. resolution and fps)
frame_width = int(front_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(front_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(front_video.get(cv2.CAP_PROP_FPS))

# Create an output video writer
out = cv2.VideoWriter('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/merged_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width*2, frame_height*2))

# Loop over the videos and merge the frames
while True:
    # Read a frame from each video
    ret1, front_frame = front_video.read()
    ret2, back_frame = back_video.read()
    ret3, left_frame = left_video.read()
    ret4, right_frame = right_video.read()

    # Check if all videos have reached the end
    if not (ret1 and ret2 and ret3 and ret4):
        break

    # Resize the frames to fit the output video
    front_frame = cv2.resize(front_frame, (frame_width, frame_height))
    back_frame = cv2.resize(back_frame, (frame_width, frame_height))
    left_frame = cv2.resize(left_frame, (frame_width, frame_height))
    right_frame = cv2.resize(right_frame, (frame_width, frame_height))

    # Combine the frames into a single image
    top_row = np.hstack((front_frame, back_frame))
    bottom_row = np.hstack((left_frame, right_frame))
    merged_frame = np.vstack((top_row, bottom_row))

    # Write the merged frame to the output video
    out.write(merged_frame)

# Release the video objects and writer
front_video.release()
back_video.release()
left_video.release()
right_video.release()
out.release()

print("Merged video saved successfully!")