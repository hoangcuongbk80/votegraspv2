import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils
from CGNL import SpatialCGNL

def decode_scores(net, end_points, num_angle_bin, num_viewpoint):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    width = net_transposed[:,:,5:6] # (batch_size, num_proposal, 1)
    end_points['width'] = width

    quality = net_transposed[:,:,6:7] # (batch_size, num_proposal, 1)
    end_points['quality'] = quality # or grasp quality or grasp score


    angle_scores = net_transposed[:,:,7:7+num_angle_bin]
    angle_residuals_normalized = net_transposed[:,:,7+num_angle_bin:7+num_angle_bin*2]
    end_points['angle_scores'] = angle_scores # Bxnum_proposalxnum_angle_bin
    end_points['angle_residuals_normalized'] = angle_residuals_normalized # Bxnum_proposalxnum_angle_bin (should be -1 to 1)
    end_points['angle_residuals'] = angle_residuals_normalized * (np.pi/num_angle_bin) # Bxnum_proposalxnum_angle_bin

    viewpoint_scores = net_transposed[:,:,7+num_angle_bin*2:7+num_angle_bin*2+num_viewpoint]
    end_points['viewpoint_scores'] = viewpoint_scores

    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_angle_bin, num_viewpoint, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_angle_bin = num_angle_bin
        self.num_viewpoint = num_viewpoint
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Grasp detection/proposal
        # Objectness-> class (2), center-> residual (3), width-> residual (1), quality(score)-> residual (1)
        # in-plane rotation-> class+residual (num_angle_bin*2), 
        # viewpoint-> class (num_viewpoint)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+1+1+num_angle_bin*2+num_viewpoint,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.sa = SpatialCGNL(128, int(128 / 2), use_scale=False, groups=4)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- CONTEXT LEARNING ---------

        feature_dim = features.shape[1]
        batch_size = features.shape[0]
        features = features.contiguous().view(batch_size, feature_dim, 16, 16)
        net = self.sa(features)
        net = net.contiguous().view(batch_size, feature_dim, self.num_proposal)

        # --------- GRASP/PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(net))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+1+1+num_angle_bin*2+num_viewpoint, num_proposal)

        end_points = decode_scores(net, end_points, self.num_angle_bin, self.num_viewpoint)
        return end_points