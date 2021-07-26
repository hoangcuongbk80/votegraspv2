import os
import numpy as np
import open3d as o3d

from graspnetAPI import GraspNet

graspnet_root = '/media/hoang/HD-PZFU3/datasets/graspnet' # ROOT PATH FOR GRASPNET

g = GraspNet(graspnet_root, camera='kinect', split='train')

sceneId = 0
annId = 0
coef_fric_thresh = 0.1
camera = 'kinect'

geometries = []
sceneGrasp = g.loadGrasp(sceneId = sceneId, annId = annId, camera = camera, format = '6d', fric_coef_thresh = coef_fric_thresh)

sceneGrasp = sceneGrasp.nms(translation_thresh = 0.1, rotation_thresh = 45.0 / 180.0 * np.pi)

scenePCD = g.loadScenePointCloud(sceneId = sceneId, camera = camera, annId = annId, align = False)
geometries.append(scenePCD)
geometries += sceneGrasp.to_open3d_geometry_list()

objectPCD = g.loadSceneModel(sceneId = sceneId, camera = camera, annId = annId, align = False)
geometries += objectPCD

o3d.visualization.draw_geometries(geometries)