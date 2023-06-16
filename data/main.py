# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import blenderproc as bproc
import argparse
import bpy
from mathutils import Matrix, Euler
import numpy as np
import cv2
import matplotlib
import os
from mathutils import Matrix

#====================================================================
#                 HELPER FUNCTIONS
#====================================================================
def set_camera_pose(cam2world_matrix, frame = None):
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)
    cam_ob = bpy.context.scene.camera
    cam_ob.matrix_world = cam2world_matrix

    bpy.context.scene.frame_end = frame + 1

    cam_ob.keyframe_insert(data_path='location', frame=frame)
    cam_ob.keyframe_insert(data_path='rotation_euler', frame=frame)

    return frame

def spherical_to_cartesian(r, theta, phi):
    return np.array([r * np.sin(theta) * np.cos(phi),
         r * np.sin(theta) * np.sin(phi),
         r * np.cos(theta)])

def relative_pose_sampler(obj):
    # Sample random rotation and apply it to the objects pose
    # the default obj.rotation_mode is XYZ Euler
    angles = (np.random.uniform(-np.pi/2,np.pi/2), np.random.uniform(-np.pi/2,np.pi/2), np.random.uniform(-np.pi/2,np.pi/2))
    obj.rotation_euler = Euler(angles)
    return angles

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()

def handle_sigterm(signum, frame):
    pass


#===================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--models-path', help="Path to the downloaded models paths, formatted like ShapeNet Core",required=True)
parser.add_argument('--output-dir', nargs='?', default="./3DIEBench", help="Path to where the final files, will be saved",required=True)
parser.add_argument('--objects', help="Path to a file containing the tuples (synset,obj) to render",required=True)
parser.add_argument('--image-size',type=int, help="image size",default=256)
parser.add_argument('--views-per-object',type=int, help="image size",default=50)
parser.add_argument('--seed',type=int, help="seed for reproducibility",default=0)

args = parser.parse_args()

np.random.seed(args.seed)

bproc.init()


items = np.load(args.objects)

print(items)
print(f"Generating for {len(items)} objects")
#====================================================================
#                 SCENE INITIALIZATION
#====================================================================

image_size = args.image_size
distance = 2.5



# Floor
bpy.ops.mesh.primitive_plane_add(size=10000,location=(0,0,-1))
floor = bpy.context.active_object #Set active object to variable
mat = bpy.data.materials.new(name="MaterialName") #set new material to variable
floor.data.materials.append(mat) #add the material to the object


# Sun (oriented so that is casts no shadows on the floor)
sun = bproc.types.Light()
sun.set_type("SUN")
sun.set_energy(1.5)
sun.blender_obj.data.angle = np.pi/2


# Spot (main lighting)
light = bproc.types.Light()
light.set_type("SPOT")
light.set_location([0, 0, 2])
# If using a white light, 100 is enough
light.set_energy(500)
light.blender_obj.data.spot_size = np.pi/8

# activate normal and distance rendering
# if activate_antialiasing is True, uses Mist pass, else uses Z pass 
bproc.renderer.enable_distance_output(activate_antialiasing=False)
# set the amount of samples, which should be used for the color rendering
bproc.renderer.set_max_amount_of_samples(100)

bproc.camera.set_resolution(image_size,image_size)

# camera
camera_theta = np.pi/4
location = spherical_to_cartesian(distance,camera_theta,np.pi/2)    
rotation_matrix = bproc.camera.rotation_from_forward_vec(np.array([0,0,0]) - location)
cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
set_camera_pose(cam2world_matrix,frame=0)

# Focal length is given in millimieters by default, so we convert it to meters
focal_length = bpy.data.cameras[0].lens/1000
print(f"focal length: {focal_length}m")

first = True
for item in items:
    synset = item[0]
    obj = item[1]
    if first :
        first = False
    else:
        bpy.data.objects.remove(bpy.context.visible_objects[-1], do_unlink=True)
    model_obj = bproc.loader.load_shapenet(args.models_path, used_synset_id=synset, used_source_id=obj)
    bb = model_obj.get_bound_box()
    bb_center = np.mean(bb,axis=0)
    model_obj.set_origin(point=bb_center)
    model_obj.set_location((0,0,0))
    
    
    os.makedirs(args.output_dir + f"/{synset}/{obj}",exist_ok=True)
    for i in range(args.views_per_object):
        if os.path.exists(args.output_dir + f"/{synset}/{obj}/latent_{i}.npy"):
            continue
        #Floor
        floor_hue = np.random.uniform(0,1)
        hsv = (floor_hue,0.6,0.6)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        floor.active_material.diffuse_color = (*rgb,1) #change color
        
        #Spot
        spot_theta = np.random.uniform(0,np.pi/4)
        spot_phi = np.random.uniform(0,2*np.pi)
        location = spherical_to_cartesian(4,spot_theta,spot_phi)    
        rotation_matrix = bproc.camera.rotation_from_forward_vec(np.array([0,0,0]) - location)
        cam2world_matrix = Matrix(bproc.math.build_transformation_mat(location, rotation_matrix))
        light.blender_obj.matrix_world = cam2world_matrix 
        
        spot_hue = np.random.uniform(0,1)
        hsv = (spot_hue ,1,0.8)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        light.set_color(rgb)
        
        #Change object pose
        yaw,pitch,roll = relative_pose_sampler(bpy.context.visible_objects[-1]) 
        model_obj.set_origin(point=[0,0,0],mode="POINT")


        latent = np.array([
                 yaw,
                 pitch,
                 roll,
                 floor_hue,
                 spot_theta,
                 spot_phi,
                 spot_hue,])

        # render the whole pipeline
        data = bproc.renderer.render()
       
        cv2.imwrite(args.output_dir + f"/{synset}/{obj}/image_{i}.jpg",data["colors"][0])
        np.save(args.output_dir + f"/{synset}/{obj}/latent_{i}.npy",latent)
