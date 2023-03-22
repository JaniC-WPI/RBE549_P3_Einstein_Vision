import bpy
import os

# Set the file paths for the model and videos
model_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Assets/Vehicles/SedanAndHatchback.blend"
front_cam_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4"
back_cam_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-back_undistort.mp4"
left_cam_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-left_repeater_undistort.mp4"
right_cam_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-right_repeater_undistort.mp4"

# Create a new scene
if bpy.data.scenes.get("Tesla Dashboard"):
    bpy.data.scenes.remove(bpy.data.scenes["Tesla Dashboard"])
    scene = bpy.data.scenes.new("Tesla Dashboard")
else: 
    scene = bpy.data.scenes.new("Tesla Dashboard")

# Open the Blender file containing the car model
bpy.ops.wm.open_mainfile(filepath=model_file)

# Create a new camera for each video
front_cam = bpy.data.cameras.new("Front Camera")
front_cam_obj = bpy.data.objects.new("Front Camera", front_cam)
front_cam_obj.location = (20.044, 18.764, -33.87)
front_cam_obj.rotation_euler = (1.046, 0.0, -1.571)
scene.collection.objects.link(front_cam_obj)

back_cam = bpy.data.cameras.new("Back Camera")
back_cam_obj = bpy.data.objects.new("Back Camera", back_cam)
back_cam_obj.location = (11.93, -40.773, -33.859)
back_cam_obj.rotation_euler = (-1.047, 0.0, -1.571)
scene.collection.objects.link(back_cam_obj)

left_cam = bpy.data.cameras.new("Left Camera")
left_cam_obj = bpy.data.objects.new("Left Camera", left_cam)
left_cam_obj.location = (13.965, -1.15, -11.888)
left_cam_obj.rotation_euler = (1.047, 0.0, -1.047)
scene.collection.objects.link(left_cam_obj)

right_cam = bpy.data.cameras.new("Right Camera")
right_cam_obj = bpy.data.objects.new("Right Camera", right_cam)
right_cam_obj.location = (8.244, -7.338, -11.906)
right_cam_obj.rotation_euler = (-1.047, 0.0, -2.094)
scene.collection.objects.link(right_cam_obj)

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 1000
# bpy.ops.sequencer.area_settings(use_split=False)

front_strip = bpy.context.scene.sequence_editor.sequences.new_movie(
    name="Front Camera Video", 
    filepath=front_cam_file, 
    channel=1, 
    frame_start=1
)

back_strip = bpy.context.scene.sequence_editor.sequences.new_movie(
    name="Back Camera Video", 
    filepath=back_cam_file, 
    channel=2, 
    frame_start=1
)

left_strip = bpy.context.scene.sequence_editor.sequences.new_movie(
    name="Left Camera Video", 
    filepath=left_cam_file, 
    channel=3, 
    frame_start=1
)

right_strip = bpy.context.scene.sequence_editor.sequences.new_movie(
    name="Right Camera Video", 
    filepath=right_cam_file, 
    channel=4, 
    frame_start=1
)

# bpy.context.area.type = 'SEQUENCE_EDITOR'

# # Load the videos into the Video Editor
# bpy.ops.sequencer.movie_strip_add(filepath=front_cam_file, channel=1, frame_start=1, name="Front Camera Video")
# bpy.ops.sequencer.movie_strip_add(filepath=back_cam_file, channel=2, frame_start=1, name="Back Camera Video")
# bpy.ops.sequencer.movie_strip_add(filepath=left_cam_file, channel=3, frame_start=1, name="Left Camera Video")
# bpy.ops.sequencer.movie_strip_add(filepath=right_cam_file, channel=4, frame_start=1, name="Right Camera Video")
# front_strip = bpy.context.scene.sequence_editor.sequences.new_movie(name="Front Camera Video", filepath=front_cam_file, channel=1, frame_start=1)
# back_strip = bpy.context.scene.sequence_editor.sequences.new_movie(name="Back Camera Video", filepath=back_cam_file, channel=2, frame_start=1)
# left_strip = bpy.context.scene.sequence_editor.sequences.new_movie(name="Left Camera Video", filepath=left_cam_file, channel=3, frame_start=1)
# right_strip = bpy.context.scene.sequence_editor.sequences.new_movie(name="Right Camera Video", filepath=right_cam_file, channel=4, frame_start=1)
# Set the dimensions and frame rate of the Video Editor
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.fps = 30
# # Set the camera for each video strip in the Video Editor
# bpy.context.scene.sequence_editor.sequences_all["front_camera_video.mp4"].clip = front_cam
# bpy.context.scene.sequence_editor.sequences_all["back_camera_video.mp4"].clip = back_cam
# bpy.context.scene.sequence_editor.sequences_all["left_camera_video.mp4"].clip = left_cam
# bpy.context.scene.sequence_editor.sequences_all["right_camera_video.mp4"].clip = right_cam
# Set the start and end frames for the Video Editor
# Render the animation
output_path = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/output.mp4"
bpy.context.scene.render.filepath = output_path
bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=False)
# bpy.ops.render.render(animation=True)
# bpy.ops.wm.quit_blender()