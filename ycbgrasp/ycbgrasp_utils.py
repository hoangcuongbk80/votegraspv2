import numpy as np
import cv2
import os
import scipy.io as sio # to load .mat files for depth points
import pc_util
import math

type2class={'007_tuna_fish_can':0, '008_pudding_box':1, '011_banana':2, '024_bowl':3, '025_mug':4, '044_flat_screwdriver':5,
            '051_large_clamp':6, '055_baseball':7, '061_foam_brick':8, '065-h_cups':9}
class2type = {type2class[t]:t for t in type2class}

class YCBObject(object):
    def __init__(self, obj_name, grasp_lines):
        self.grasps = []
        self.classname = obj_name
        for line in grasp_lines:
            data = line.split(' ')
            data[1:] = [float(x) for x in data[1:]]
            grasp = [data[1],data[2],data[3],data[4],data[5],data[6], data[7]] # grasp_position(3), viewpoint, angle_angle, quality, width
            self.grasps.append(grasp)

def load_pointcloud(pc_filename):
    pointcloud = pc_util.read_xyzrgb_ply(pc_filename)
    return pointcloud

def load_label(grasp_filename, num_grasp):
    lines = [line.rstrip() for line in open(grasp_filename)]
    grasp_lines = []
    obj_name = ''
    objects = []
    for line in lines[1:]:
        data = line.split(' ')
        if data[0] != obj_name:
            if obj_name != '':
                obj = YCBObject(obj_name, grasp_lines)
                objects.append(obj)
            obj_name = data[0]
            grasp_lines = []
            grasp_lines.append(line)
        else:
            if len(grasp_lines) < num_grasp:
                grasp_lines.append(line)
    return objects

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, angle):
    R = rotz(-1*angle)
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    print(box3d_roi_inds)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def get_object_points(pc, object_name):
    rgb=pc[:,3]
    id = type2class[object_name] + 1
    inds=rgb[:]==id
    return pc[inds,:], inds
