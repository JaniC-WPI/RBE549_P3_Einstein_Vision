import bpy
import cv2
import numpy as np

# Load the image and convert it to a texture
img_path = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene2/images31/test_img.jpg"
img = cv2.imread(img_path)
h, w, c = img.shape
tex = bpy.data.textures.new("RoadTexture", type="IMAGE")
img_data = bpy.data.images.new("RoadImage", w, h)
img_data.pixels = img.reshape(-1)
tex.image = img_data

# Create a new material and assign the texture to it
mat = bpy.data.materials.new("RoadMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
tex_node = nodes.new(type="ShaderNodeTexImage")
tex_node.image = img_data
principled_node = nodes.get("Principled BSDF")
mat.node_tree.links.new(tex_node.outputs[0], principled_node.inputs[0])

# Create a plane and assign the material to it
plane = bpy.data.objects.new("RoadPlane", bpy.data.meshes.new("RoadMesh"))
bpy.context.scene.collection.objects.link(plane)
bpy.context.view_layer.objects.active = plane
plane.data.vertices.add(4)
plane.data.vertices[0].co = (-w/2, -h/2, 0)
plane.data.vertices[1].co = (w/2, -h/2, 0)
plane.data.vertices[2].co = (w/2, h/2, 0)
plane.data.vertices[3].co = (-w/2, h/2, 0)
plane.data.polygons.add(1)
plane.data.polygons[0].vertices = [0, 1, 2, 3]
plane.data.uv_layers.new()
plane.data.uv_layers[0].data[0].uv = (0, 0)
plane.data.uv_layers[0].data[1].uv = (1, 0)
plane.data.uv_layers[0].data[2].uv = (1, 1)
plane.data.uv_layers[0].data[3].uv = (0, 1)
plane.data.materials.append(mat)

# Add a camera and position it above the plane
cam = bpy.data.cameras.new("Camera")
cam_ob = bpy.data.objects.new("Camera", cam)
bpy.context.scene.collection.objects.link(cam_ob)
bpy.context.view_layer.objects.active = cam_ob
cam_ob.location = (0, 0, 20)
cam_ob.rotation_euler = (np.radians(-90), 0, 0)

# Add a lamp
lamp_data = bpy.data.lights.new(name="Lamp", type='POINT')
lamp_ob = bpy.data.objects.new(name="Lamp", object_data=lamp_data)
bpy.context.scene.collection.objects.link(lamp_ob)
bpy.context.view_layer.objects.active = lamp_ob
lamp_ob.location = (0, -20, 20)
lamp_data.energy = 1000

# Render the scene
render = bpy.context.scene.render
render.engine = 'BLENDER_EEVEE'
render.resolution_x = w
render.resolution_y = h
render.resolution_percentage = 100
render.use_overwrite = True
bpy.ops.render.render(write_still=True)