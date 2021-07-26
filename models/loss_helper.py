import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.15
NEAR_THRESHOLD = 0.06
GT_VOTE_FACTOR = 10 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for grasp points of the object.
        Each seed point may vote for multiple grasps
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,30 and 30 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,GT_VOTE_FACTOR)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the grasps/proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_grasp-1]
    """ 
    # Associate proposal and GT grasps by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred grasp center is within NEAR_THRESHOLD of any GT grasp
    # objectness_mask: 0 if pred grasp center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_grasp_loss(end_points, config):
    """ Compute grasp loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        angle_cls_loss
        angle_reg_loss
        viewpoint_cls_loss
    """

    num_angle_bin = config.num_angle_bin
    num_viewpoint = config.num_viewpoint

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    grasp_label_mask = end_points['grasp_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*grasp_label_mask)/(torch.sum(grasp_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute angle loss (in-plane rotation)
    angle_class_label = torch.gather(end_points['angle_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_angle_class = nn.CrossEntropyLoss(reduction='none')
    angle_class_loss = criterion_angle_class(end_points['angle_scores'].transpose(2,1), angle_class_label) # (B,K)
    angle_class_loss = torch.sum(angle_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    angle_residual_label = torch.gather(end_points['angle_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    angle_residual_normalized_label = angle_residual_label / (np.pi/num_angle_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    angle_label_one_hot = torch.cuda.FloatTensor(batch_size, angle_class_label.shape[1], num_angle_bin).zero_()
    angle_label_one_hot.scatter_(2, angle_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_angle_bin)
    angle_residual_normalized_loss = huber_loss(torch.sum(end_points['angle_residuals_normalized']*angle_label_one_hot, -1) - angle_residual_normalized_label, delta=1.0) # (B,K)
    cuong = torch.sum(end_points['angle_residuals_normalized']*angle_label_one_hot, -1)
    angle_residual_normalized_loss = torch.sum(angle_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute width loss
    gt_width = torch.gather(end_points['width_label'], 1, object_assignment) # select (B,K) from (B,K2)
    width_loss = huber_loss(torch.sum(end_points['width'], -1) - gt_width, delta=1.0)
    width_loss = torch.sum(width_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute quality loss
    gt_quality = torch.gather(end_points['quality_label'], 1, object_assignment) # select (B,K) from (B,K2)
    quality_loss = huber_loss(torch.sum(end_points['quality'], -1) - gt_quality, delta=1.0)
    quality_loss = torch.sum(quality_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute viewpoint loss
    viewpoint_class_labell = torch.gather(end_points['viewpoint_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_viewpoint_class = nn.CrossEntropyLoss(reduction='none')
    viewpoint_class_loss = criterion_viewpoint_class(end_points['viewpoint_scores'].transpose(2,1), viewpoint_class_labell) # (B,K)
    viewpoint_class_loss = torch.sum(viewpoint_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, width_loss, quality_loss, angle_class_loss, angle_residual_normalized_loss, viewpoint_class_loss

def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                angle_scores, angle_residuals_normalized,
                viewpoint_scores,
                center_label,
                angle_class_label, angle_residual_label,
                viewpoint_class_labell,
                grasp_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # grasp loss
    center_loss, width_loss, quality_loss, angle_cls_loss, angle_reg_loss, viewpoint_cls_loss = \
        compute_grasp_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['width_loss'] = width_loss
    end_points['quality_loss'] = quality_loss
    end_points['angle_cls_loss'] = angle_cls_loss
    end_points['angle_reg_loss'] = angle_reg_loss
    end_points['viewpoint_cls_loss'] = viewpoint_cls_loss
    grasp_loss = center_loss + quality_loss + width_loss + 0.1*angle_cls_loss + angle_reg_loss + 0.1*viewpoint_cls_loss
    end_points['grasp_loss'] = grasp_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + grasp_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points
