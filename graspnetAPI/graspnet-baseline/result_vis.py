import os
import sys
import numpy as np

import open3d as o3d
from PIL import Image
import scipy.io as scio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnetAPI import GraspNet, Grasp, GraspGroup
from data_utils import CameraInfo, create_point_cloud_from_depth_image


data_dir = os.path.join(ROOT_DIR, 'doc/example_data')
num_point = 20000

# load data
color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']

# generate cloud
camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

 # get valid points
mask = (workspace_mask & (depth > 0))

cloud_masked = cloud[mask]
color_masked = color[mask]

# sample points
if len(cloud_masked) >= num_point:
    idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
else:
    idxs1 = np.arange(len(cloud_masked))
    idxs2 = np.random.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
    idxs = np.concatenate([idxs1, idxs2], axis=0)
cloud_sampled = cloud_masked[idxs]
color_sampled = color_masked[idxs]

# convert data
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

gg = GraspGroup().from_npy(os.path.join(ROOT_DIR, 'doc/example_data/result.npy'))
gg = gg[:100]
grippers = gg.to_open3d_geometry_list()
o3d.visualization.draw_geometries([cloud, *grippers])