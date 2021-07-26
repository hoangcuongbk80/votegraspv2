import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class ycbgraspDatasetConfig(object):
    def __init__(self):
        self.num_class = 88
        self.num_angle_bin = 12
        self.num_viewpoint = 300

        self.type2class={'007_tuna_fish_can':0, '008_pudding_box':1, '011_banana':2, '024_bowl':3, '025_mug':4,
                        '044_flat_screwdriver':5, '051_large_clamp':6, '055_baseball':7, '061_foam_brick':8, '065-h_cups':9}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type2onehotclass={'007_tuna_fish_can':0, '008_pudding_box':1, '011_banana':2, '024_bowl':3, '025_mug':4,
                            '044_flat_screwdriver':5, '051_large_clamp':6, '055_baseball':7, '061_foam_brick':8, '065-h_cups':9}

    
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_angle_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_angle_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2grasp(self, sem_cls, center, viewpoint_class, angle_class, angle_residual, quality, width):
        angle = self.class2angle(angle_class, angle_residual) * 180/np.pi
        object_name = self.class2type[int(sem_cls)]
        grasp = []
        grasp.append(object_name)
        grasp.append(center[0])
        grasp.append(center[1])
        grasp.append(center[2])
        grasp.append(viewpoint_class)
        grasp.append(angle)
        grasp.append(quality)
        grasp.append(width)
        return grasp
