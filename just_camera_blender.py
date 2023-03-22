import bpy

# Load the video files as textures
# front_texture = bpy.data.textures.load("/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4")
# back_texture = bpy.data.textures.load("/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-back_undistort.mp4")
# left_texture = bpy.data.textures.load("/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-left_repeater_undistort.mp4")
# right_texture = bpy.data.textures.load("/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-right_repeater_undistort.mp4")

# Load the video files as images
bpy.ops.image.open(filepath="/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4")
front_image = bpy.data.images[-1]
bpy.ops.image.open(filepath="/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-back_undistort.mp4")
back_image = bpy.data.images[-1]
bpy.ops.image.open(filepath="/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-left_repeater_undistort.mp4")
left_image = bpy.data.images[-1]
bpy.ops.image.open(filepath="/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-right_repeater_undistort.mp4")
right_image = bpy.data.images[-1]

# Create a texture from each image
front_texture = bpy.data.textures.new('FrontTexture', type='IMAGE')
front_texture.image = front_image
back_texture = bpy.data.textures.new('BackTexture', type='IMAGE')
back_texture.image = back_image
left_texture = bpy.data.textures.new('LeftTexture', type='IMAGE')
left_texture.image = left_image
right_texture = bpy.data.textures.new('RightTexture', type='IMAGE')
right_texture.image = right_image

# Set the texture properties
for texture in [front_texture, back_texture, left_texture, right_texture]:
    texture.use_preview_alpha = False
    texture.use_alpha = False
    texture.use_mipmap = False
    texture.filter_type = 'BOX'
    texture.extension = 'REPEAT'
    texture.frame_start = 1
    texture.frame_end = front_image.frame_duration
    texture.frame_method = 'OFFSET'
    texture.image_user.frame_offset = 0
    texture.image.colorspace_settings.name = 'sRGB'

# Create a plane and assign the videos as textures
plane = bpy.data.objects.new('Plane', bpy.data.meshes.new('PlaneMesh'))
plane.data.materials.append(bpy.data.materials.new(name='CombinedMaterial'))

# Create a material that combines the four videos into one
combined_material = bpy.data.materials.new(name='CombinedMaterial')
combined_material.use_nodes = True
nodes = combined_material.node_tree.nodes
links = combined_material.node_tree.links
diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
texture_node_front = nodes.new('ShaderNodeTexImage')
texture_node_back = nodes.new('ShaderNodeTexImage')
texture_node_left = nodes.new('ShaderNodeTexImage')
texture_node_right = nodes.new('ShaderNodeTexImage')
group_node = nodes.new('ShaderNodeGroup')
group_node.node_tree = bpy.data.node_groups.new('VideoTextureGroup')
group_node.inputs[0].default_value = front_texture
group_node.inputs[1].default_value = back_texture
group_node.inputs[2].default_value = left_texture
group_node.inputs[3].default_value = right_texture
nodes.remove(diffuse_node)
links.new(group_node.outputs[0], combined_material.node_tree.nodes["Material Output"].inputs[0])

plane.data.materials[0] = combined_material
plane.scale = (2, 2, 2)

# Add a camera and position it
camera = bpy.data.cameras.new('Camera')
camera_obj = bpy.data.objects.new('Camera', camera)
camera_obj.location = (0, -10, 0)
camera_obj.rotation_euler = (0, 0, 0)
bpy.context.scene.camera = camera_obj

# Add lights to the scene
light_data = bpy.data.lights.new('Light', type='POINT')
light_data.color = (1.0, 1.0, 1.0)
light_data.energy = 1000.0
light = bpy.data.objects.new('Light', light_data)
light.location = (0, 0, 10)
bpy.context.scene.collection.objects.link(light)

# Set the output properties for the final video
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.fps = 30
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.image_settings.codec = 'H264'
bpy.context.scene.render.filepath = 'merged_video.mp4'

# Render the scene
bpy.ops.render.render(write_still=False)

print("Merged video saved successfully!")
bpy.ops.render.render(animation=True)