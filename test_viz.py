import bpy
import cv2

# Set up the cameras
front_cam = bpy.data.objects.new('Front Cam', bpy.data.cameras.new('Front Cam'))
front_cam.location = (-10, 0, 0)
front_cam.rotation_euler = (0, 0, 0)

back_cam = bpy.data.objects.new('Back Cam', bpy.data.cameras.new('Back Cam'))
back_cam.location = (10, 0, 0)
back_cam.rotation_euler = (0, 0, 180)

left_cam = bpy.data.objects.new('Left Cam', bpy.data.cameras.new('Left Cam'))
left_cam.location = (0, -10, 0)
left_cam.rotation_euler = (0, 0, 90)

right_cam = bpy.data.objects.new('Right Cam', bpy.data.cameras.new('Right Cam'))
right_cam.location = (0, 10, 0)
right_cam.rotation_euler = (0, 0, -90)

# Set up the scene
scene = bpy.context.scene
scene.camera = front_cam
scene.collection.objects.link(front_cam)
scene.collection.objects.link(back_cam)
scene.collection.objects.link(left_cam)
scene.collection.objects.link(right_cam)

# Read the videos from local file
front_video = cv2.VideoCapture('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4')
back_video = cv2.VideoCapture('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-back_undistort.mp4')
left_video = cv2.VideoCapture('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-left_repeater_undistort.mp4')
right_video = cv2.VideoCapture('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-right_repeater_undistort.mp4')

# Render the scene from each camera
for cam, video in zip([front_cam, back_cam, left_cam, right_cam], [front_video, back_video, left_video, right_video]):
    scene.camera = cam
    ret, frame = video.read()
    while ret:
        # Set the video frame as background image for the scene
        img = bpy.data.images.new(name="Video Frame", width=frame.shape[1], height=frame.shape[0])
        img.pixels = frame.flatten()
#        scene.world.node_tree.nodes['Background'].inputs[1].default_value[0] = img.pixels[0]
#        scene.world.node_tree.nodes['Background'].inputs[1].default_value[1] = img.pixels[1]
#        scene.world.node_tree.nodes['Background'].inputs[1].default_value[2] = img.pixels[2]
#        scene.world.node_tree.nodes['Background'].inputs[1].default_value[3] = 1.0
        rgba = (*img.pixels[:3], 1.0)
        scene.world.node_tree.nodes['Background'].inputs[1].default_value = rgba


        # Render the scene to the output file
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.filepath = '/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/output.jpg'
        bpy.ops.render.render(write_still=True)

        # Read the next frame
        ret, frame = video.read()

    # Release the video when finished
    video.release()
