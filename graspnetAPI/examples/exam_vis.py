__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for visualization.
# change the graspnet_root path

####################################################################
graspnet_root = '/media/hoang/HD-PZFU3/datasets/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

from graspnetAPI import GraspNet

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# show object grasps
#g.showObjGrasp(objIds = 0, numGrasp=10, show=True)

# show 6d poses
#g.show6DPose(sceneIds = 0, show = True)

# show scene rectangle grasps
#g.showSceneGrasp(sceneId = 0, camera = 'realsense', annId = 0, format = 'rect', numGrasp = 20)

# show scene 6d grasps(You may need to wait several minutes)
g.showSceneGrasp(sceneId = 4, numGrasp=100, coef_fric_thresh = 0.1, camera = 'kinect', annId = 2, format = '6d')