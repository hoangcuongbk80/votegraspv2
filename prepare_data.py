import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
from tqdm import tqdm
from PIL import Image

from graspnetAPI import GraspNet, GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points

root = "/media/hoang/HD-PZFU3/datasets/graspnet"
#root = "/graspnet"

display = True

num_points = 50000
num_grasp = 10
remove_outlier = True
valid_obj_idxs = []
grasp_labels = {}
camera = 'kinect'
collision_labels = {}

sceneIds = list( range(100)) #100
sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in sceneIds]
        
colorpath = []
depthpath = []
labelpath = []
metapath = []
scenename = []
frameid = []
geometries = []

for x in tqdm(sceneIds, desc = 'Loading data path and collision labels...'):
    for img_num in range(256): #256
        colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
        depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
        labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
        metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
        scenename.append(x.strip())
        frameid.append(img_num)
    
        collision_labels_f = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
        collision_labels[x.strip()] = {}
        for i in range(len(collision_labels_f)):
            collision_labels[x.strip()][i] = collision_labels_f['arr_{}'.format(i)]

def load_grasp_labels(root):
    obj_names = list(range(0,88)) #88
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if obj_name == 18: continue
        valid_obj_idxs.append(obj_name) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(obj_name).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32))

    return valid_obj_idxs, grasp_labels

def get_pointcloud(index):
    color = np.array(Image.open(colorpath[index]), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depthpath[index]))
    seg = np.array(Image.open(labelpath[index]))
    meta = scio.loadmat(metapath[index])
    scene = scenename[index]
    try:
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        poses = meta['poses']
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
        print(scene)
    camera_info = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    seg_mask = (seg > 0)
    if remove_outlier:
            camera_poses = np.load(os.path.join(root, 'scenes', scene, camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(root, 'scenes', scene, camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
    else:
        mask = depth_mask

    cloud_masked = cloud[mask]
    color_masked = color[mask]
    seg_masked = seg[mask]

    # sample points
    if len(cloud_masked) >= num_points:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_points-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    seg_sampled = seg_masked[idxs]
    #objectness_label = seg_sampled.copy()
    #objectness_label[objectness_label>1] = 1
    return cloud_sampled, color_sampled, seg_sampled


def get_grasp_label(index):
    graspnet_root = root # ROOT PATH FOR GRASPNET
    g = GraspNet(graspnet_root, camera='kinect', split='train')

    sceneId = int(scenename[index][-4:])
    annId = frameid[index]
    coef_fric_thresh = 0.2
    camera = 'kinect'

    sceneGrasp = g.loadGraspv2(sceneId = sceneId, annId = annId, camera = camera, valid_obj_idxs = valid_obj_idxs, \
                grasp_labels = grasp_labels, collision_labels = collision_labels, fric_coef_thresh = coef_fric_thresh)

    return sceneGrasp

def compute_votes(sceneGrasp, cloud_sampled, color_sampled, seg_sampled):
    obj_ids = np.unique(seg_sampled)
    grasp_group_array = np.zeros((0, 22), dtype=np.float64)

    grasp_list = []
    N = cloud_sampled.shape[0]
    point_votes = np.zeros((N,31)) # 10 votes and 1 vote mask 10*3+1 
    point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
    indices = np.arange(N)

    for id in obj_ids:
        obj_idx = id-1 #here align label png with object ids
        if obj_idx not in valid_obj_idxs:
                continue
        grasp_mask = (sceneGrasp.grasp_group_array[:,16] == obj_idx)
        objectGrasp = GraspGroup(sceneGrasp.grasp_group_array[grasp_mask])

        trans_thresh = 0.1
        rot_thresh = 45.0
        nms_grasp = objectGrasp.nms(translation_thresh = trans_thresh, rotation_thresh = rot_thresh / 180.0 * np.pi)
        while len(nms_grasp.grasp_group_array) < num_grasp and trans_thresh > 0 and rot_thresh > 0:
            trans_thresh = trans_thresh - 0.2
            rot_thresh = rot_thresh - 10.0
            nms_grasp = objectGrasp.nms(translation_thresh = trans_thresh, rotation_thresh = rot_thresh / 180.0 * np.pi)
        if len(nms_grasp.grasp_group_array) < num_grasp:
            continue
        score_sorted_grasps = nms_grasp.grasp_group_array[nms_grasp.grasp_group_array[:, 0].argsort()]
        objectGrasp.grasp_group_array = score_sorted_grasps[::-1] #Reverse the sorted array
        grasps = objectGrasp.grasp_group_array[:10]
        grasp_group_array = np.concatenate((grasp_group_array, grasps))

        # Add grasp
        for grp in grasps:
            grasp = np.zeros((8))
            grasp[0:3] = np.array([grp[13], grp[14], grp[15]]) # grasp_position
            grasp[3] = grp[21] # viewpoint
            grasp[4] = grp[17] # angle
            grasp[5] = grp[0] # quality
            grasp[5] = grp[1] # width
            grasp[7] = id # semantic class id
            grasp_list.append(grasp)

        inds = seg_sampled==id
        object_pc = cloud_sampled[inds]
        if len(object_pc) < 200:
            continue

        # Assign first dimension to indicate it belongs an object
        point_votes[inds,0] = 1
        for grasp_idx, grp in enumerate(grasps):
            grasp_position = np.array([grp[0], grp[1], grp[2]])
            # Add the votes (all 0 if the point is not in any object's OBB)
            votes = np.expand_dims(grasp_position,0) - object_pc[:,0:3]
            sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
            for i in range(len(sparse_inds)):
                j = sparse_inds[i]
                point_votes[j, int(grasp_idx*3+1):int((grasp_idx+1)*3+1)] = votes[i,:]

    if len(grasp_list)==0:
        final_grasps = np.zeros((0,8))
    else:
        final_grasps = np.vstack(grasp_list) # (K,8)

    sceneGrasp.grasp_group_array = grasp_group_array

    return sceneGrasp, point_votes, final_grasps

def extract_data(data_dir, idx_filename, output_folder):
    
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        cloud_sampled, color_sampled, seg_sampled = get_pointcloud(data_idx)        
        sceneGrasp = get_grasp_label(data_idx)
        sceneGrasp, point_votes, grasps = compute_votes(sceneGrasp, cloud_sampled, color_sampled, seg_sampled)

        np.savez_compressed(os.path.join(output_folder,'%06d_pc.npz'%(data_idx)), pc=cloud_sampled)
        np.savez_compressed(os.path.join(output_folder, '%06d_votes.npz'%(data_idx)), point_votes = point_votes)
        np.save(os.path.join(output_folder, '%06d_grasp.npy'%(data_idx)), grasps)
        
        if display:
            geometries = []
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cloud_sampled)
            cloud.colors = o3d.utility.Vector3dVector(color_sampled)
            geometries.append(cloud)
            geometries += sceneGrasp.to_open3d_geometry_list()
            o3d.visualization.draw_geometries(geometries)

if __name__=='__main__':
        idxs = np.array(range(0,len(depthpath)))
        num_train = (int)(0.75*len(idxs))
        np.random.seed(0)
        np.random.shuffle(idxs)
        
        DATA_DIR = os.path.join(root, 'data')
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        
        np.savetxt(os.path.join(root, 'data', 'train_data_idx.txt'), idxs[:num_train], fmt='%i')
        np.savetxt(os.path.join(root, 'data', 'val_data_idx.txt'), idxs[num_train:], fmt='%i')
        
        valid_obj_idxs, grasp_labels = load_grasp_labels(root)
        
        extract_data(DATA_DIR, os.path.join(DATA_DIR, 'train_data_idx.txt'), output_folder = os.path.join(DATA_DIR, 'train'))
        extract_data(DATA_DIR, os.path.join(DATA_DIR, 'val_data_idx.txt'), output_folder = os.path.join(DATA_DIR, 'val'))