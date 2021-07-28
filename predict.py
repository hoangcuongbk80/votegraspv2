""" Predict/generate grasps from a point cloud using VoteGrasp.
sample usage: python predict.py --input points.ply
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='points.ply', help='Input: pointcloud [default: points.ply]')
parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 50000]')
FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply, read_xyzrgb_ply

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

if __name__=='__main__':
    
    # Set file paths and dataset config
    dump_dir = os.path.join(BASE_DIR, 'pred_result')
    log_dir = os.path.join(BASE_DIR, 'log') 

    sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
    from dataset import DC # dataset config
    checkpoint_path = os.path.join(log_dir, 'checkpoint.tar') # trained model path
    pc_path = FLAGS.input


    # Init the model and optimzier
    MODEL = importlib.import_module('votegrasp') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteGrasp(num_proposal=256, input_feature_dim=1, vote_factor=10,
        sampling='seed_fps',
        num_angle_bin=DC.num_angle_bin,
        num_viewpoint=DC.num_viewpoint).to(device)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)
    # point_cloud = read_ply(pc_path)
    point_cloud = read_xyzrgb_ply(pc_path)
    pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))
   
    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))
    end_points['point_clouds'] = inputs['point_clouds']
  
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    MODEL.dump_results(end_points, dump_dir, DC, True)
    print('Dumped detection results to folder %s'%(dump_dir))
