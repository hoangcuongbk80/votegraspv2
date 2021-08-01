import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

DUMP_CONF_THRESH = 0.5 # Dump grasps with grasp prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    '''
    **Input:**

    - towards: numpy array towards vectors with shape (n, 3).

    - angle: numpy array of in-plane rotations (n, ).

    **Output:**

    - numpy array of the rotation matrix with shape (n, 3, 3).
    '''
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)
    axis_y[mask_y] = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R1 = R1.reshape([-1,3,3])
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32)

def dump_results(end_points, dump_dir, config, inference_switch=False):

    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
        vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    pred_angle_class = torch.argmax(end_points['angle_scores'], -1) # B,num_proposal
    pred_angle_residual = torch.gather(end_points['angle_residuals'], 2, pred_angle_class.unsqueeze(-1)) # B,num_proposal,1
    pred_angle_class = pred_angle_class.detach().cpu().numpy() # B,num_proposal
    pred_angle_residual = pred_angle_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_viewpoint_class = torch.argmax(end_points['viewpoint_scores'], -1) # B,num_proposal

    pred_quality = end_points['quality'].detach().cpu().numpy() # B,num_proposal
    pred_width = end_points['width'].detach().cpu().numpy() # B,num_proposal

    print("pred_angle_class: ", pred_angle_class)
    print("pred_viewpoint_class: ", pred_viewpoint_class)

    batch_viewpoint_params_to_matrix(pred_angle_class, pred_viewpoint_class)

    # OTHERS
    #pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0

    save_grasp_file = os.path.join(dump_dir, 'pred_grasps.txt')
    f = open(save_grasp_file, "w")
    f.write("object x y z viewpoint angle quality width\n")

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        if 'vote_xyz' in end_points:
            pc_util.write_ply(end_points['vote_xyz'][i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(idx_beg+i)))



        # Dump predicted grasps
        """ if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            for j in range(num_proposal):
                grasp = config.param2grasp(pred_center[i,j,0:3], pred_viewpoint_class[i,j],
                            pred_angle_class[i,j], pred_angle_residual[i,j], pred_quality[i,j,0], pred_width[i,j,0])
                f.write(grasp[0])
                f.write(' ')             
                for ite in grasp[1:]:
                    str_num = '{:.6f}'.format(ite)
                    f.write(str_num)
                    f.write(' ')
                f.write("\n") """

    f.close()