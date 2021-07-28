import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))
import pc_util
import votegrasp_utils

parser = argparse.ArgumentParser()
parser.add_argument('--viz', action='store_true', help='Run data visualization.')
parser.add_argument('--gen_data', action='store_true', help='Generate training dataset.')
parser.add_argument('--num_sample', type=int, default=90000, help='Number of samples [default: 90000]')
parser.add_argument('--num_grasp', type=int, default=10, help='Number of grasps per objects [default: 10]')
parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 50000]')

args = parser.parse_args()

DEFAULT_TYPE_WHITELIST = ['007_tuna_fish_can', '008_pudding_box', '011_banana', '024_bowl', '025_mug', '044_flat_screwdriver',
                            '051_large_clamp', '055_baseball', '061_foam_brick', '065-h_cups']

class ycb_object(object):
    ''' Load and parse object data '''
    def __init__(self, data_dir):

        self.pointcloud_dir = os.path.join(data_dir, 'pointcloud')
        self.grasp_dir = os.path.join(data_dir, 'grasp')
        self.num_samples = args.num_sample
        self.num_grasps = args.num_grasp
        
    def __len__(self):
        return self.num_samples

    def get_pointcloud(self, idx):
        pc_filename = os.path.join(self.pointcloud_dir, '%d.ply'%(idx))
        print(pc_filename)
        return votegrasp_utils.load_pointcloud(pc_filename)

    def get_label_objects(self, idx): 
        grasp_filename = os.path.join(self.grasp_dir, '%d.txt'%(idx))
        print(grasp_filename)
        return votegrasp_utils.load_label(grasp_filename, self.num_grasps)

def data_viz(data_dir, dump_dir=os.path.join(BASE_DIR, 'data_viz_dump')):
    ''' Examine and visualize votegrasp dataset. '''
    ycb = ycb_object(data_dir)
    idxs = np.array(range(0,len(ycb)))

    if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

    for idx in range(len(ycb)):
        if idx%10:
            continue
        data_idx = idxs[idx]
        print('data index: ', data_idx)
        pc = ycb.get_pointcloud(data_idx)
        pc=pc[:,0:3]
        pc = pc_util.random_sampling(pc, args.num_point)
        pc_util.write_ply(pc, os.path.join(dump_dir, str(idx) + '_pc.ply'))
        
    print('Complete!')
    
def extract_votegrasp_data(data_dir, idx_filename, output_folder, num_point=20000,
    type_whitelist=DEFAULT_TYPE_WHITELIST):
    
    dataset = ycb_object(data_dir)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        objects = dataset.get_label_objects(data_idx)

        # Save pointcloud
        pc = dataset.get_pointcloud(data_idx)
        xyz_pc=pc[:,0:3]
        np.savez_compressed(os.path.join(output_folder,'%06d_pc.npz'%(data_idx)), pc=xyz_pc)
        # Save grasps and votes
        grasp_list = []
        N = pc.shape[0]
        point_votes = np.zeros((N,31)) # 10 votes and 1 vote mask 10*3+1 
        point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
        indices = np.arange(N)
        for obj in objects:
            object_pc, inds=votegrasp_utils.get_object_points(pc, obj.classname)
            
            if len(object_pc) < 300:
                continue
            # Add grasp
            for grp in obj.grasps:
                grasp = np.zeros((8))
                grasp[0:3] = np.array([grp[0], grp[1], grp[2]]) # grasp_position
                grasp[3] = grp[3] # viewpoint
                grasp[4] = grp[4] # angle
                grasp[5] = grp[5] # quality
                grasp[5] = grp[6] # width
                grasp[7] = votegrasp_utils.type2class[obj.classname] # semantic class id
                grasp_list.append(grasp)
            
            # Assign first dimension to indicate it belongs an object
            point_votes[inds,0] = 1
            for grasp_idx, grp in enumerate(obj.grasps):
                grasp_position = np.array([grp[0], grp[1], grp[2]])
                # Add the votes (all 0 if the point is not in any object's OBB)
                votes = np.expand_dims(grasp_position,0) - object_pc[:,0:3]
                sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
                for i in range(len(sparse_inds)):
                    j = sparse_inds[i]
                    point_votes[j, int(grasp_idx*3+1):int((grasp_idx+1)*3+1)] = votes[i,:]

        np.savez_compressed(os.path.join(output_folder, '%06d_votes.npz'%(data_idx)), point_votes = point_votes)
        if len(grasp_list)==0:
            grasps = np.zeros((0,8))
        else:
            grasps = np.vstack(grasp_list) # (K,8)
        np.save(os.path.join(output_folder, '%06d_grasp.npy'%(data_idx)), grasps)

    return 0

    
if __name__=='__main__':
    
    if args.viz:
        data_viz(os.path.join(BASE_DIR, 'data'))
        exit()

    if args.gen_data:
        idxs = np.array(range(0,args.num_sample))
        np.random.seed(0)
        np.random.shuffle(idxs)
        np.savetxt(os.path.join(BASE_DIR, 'data', 'train_data_idx.txt'), idxs[:75000], fmt='%i')
        np.savetxt(os.path.join(BASE_DIR, 'data', 'val_data_idx.txt'), idxs[15000:], fmt='%i')
        
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        extract_votegrasp_data(DATA_DIR, os.path.join(DATA_DIR, 'train_data_idx.txt'),
            output_folder = os.path.join(DATA_DIR, 'train'), num_point=50000)
        extract_votegrasp_data(DATA_DIR, os.path.join(DATA_DIR, 'val_data_idx.txt'),
            output_folder = os.path.join(DATA_DIR, 'val'), num_point=50000)