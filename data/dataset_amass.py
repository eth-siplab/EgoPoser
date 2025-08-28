'''
# --------------------------------------------
# dataloader for AMASS dataset, dapted from AvatarPoser (ECCV 2022)
# --------------------------------------------
# EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere (ECCV2024)
# https://github.com/eth-siplab/EgoPoser
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import torch
import numpy as np
import os
from torch.utils.data import Dataset
import random
import glob
import pickle

class AMASS_Dataset(Dataset):
    """Motion Capture dataset"""

    def __init__(self, opt):
        self.opt = opt
        self.window_size = opt['window_size']
        self.num_input = opt['num_input']
        self.fov_h = opt['fov_h']
        self.fov_v = opt['fov_v']
        self.batch_size = opt['dataloader_batch_size']
        dataroot_list = opt['dataroot']
        self.offset= opt['offset']
        self.full_hand_visibility = opt['full_hand_visibility']
        self.sample_stride = opt['sample_stride']


        if self.opt['phase'] == 'train':
            self.filename_list = []
            for dataroot in dataroot_list:
                filenames_train = os.path.join(dataroot, '*.pkl')
                self.filename_list += glob.glob(filenames_train)
        else:
            self.filename_list = []
            for dataroot in dataroot_list:
                filenames_test = os.path.join(dataroot, '*.pkl')
                self.filename_list += glob.glob(filenames_test)
            self.filename_list.sort()
            print('-------------------------------number of test data is {}'.format(len(self.filename_list)))

    def __len__(self):

        return max(len(self.filename_list), self.batch_size)


    # field of view modeling

    def in_fov(self, points, fov_h, fov_v):

        in_horizontor = np.arccos(points[:, 2]/np.sqrt(np.square(points[:, 0])+np.square(points[:, 2]))) <= np.deg2rad(fov_h/2)
        in_vertical = np.arccos(points[:, 2]/np.sqrt(np.square(points[:, 1])+np.square(points[:, 2]))) <= np.deg2rad(fov_v/2)
        in_fov = in_horizontor & in_vertical
        return in_fov.squeeze()

    def __getitem__(self, idx):

        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if self.opt['phase'] == 'train':
            while data['rotation_local_full_gt_list'].shape[0] <= self.window_size:
                idx = random.randint(0,idx)
                filename = self.filename_list[idx]
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

        rotation_local_full_gt_list = data['rotation_local_full_gt_list']
        hmd_position_global_full_gt_list = data['hmd_position_global_full_gt_list']
        head_global_trans_list = data['head_global_trans_list']
        root_trans = data['trans']
        betas = data['betas'] if 'betas' in data.keys() else None

        if self.opt['phase'] == 'train':
            
            frame = np.random.randint(hmd_position_global_full_gt_list.shape[0] - self.window_size)
            input_hmd  = hmd_position_global_full_gt_list[frame:frame + self.window_size,...].reshape(self.window_size, -1).float()
            output_gt = rotation_local_full_gt_list[frame:frame + self.window_size,...].float()

            head_global_trans_inv_list = torch.inverse(head_global_trans_list)[frame:frame + self.window_size,...]

            position_lefthand_gt_world = input_hmd[:,36+3:36+6]
            position_lefthand_gt_world_aug = torch.cat([position_lefthand_gt_world, torch.ones(position_lefthand_gt_world.shape[0],1)],dim=1)

            position_lefthand_head = torch.matmul(head_global_trans_inv_list,position_lefthand_gt_world_aug.unsqueeze(-1)).squeeze()[:,:3]

            position_righthand_gt_world = input_hmd[:,36+6:36+9]
            position_righthand_gt_world_aug = torch.cat([position_righthand_gt_world, torch.ones(position_righthand_gt_world.shape[0],1)],dim=1)
            position_righthand_head = torch.matmul(head_global_trans_inv_list,position_righthand_gt_world_aug.unsqueeze(-1)).squeeze()[:,:3]


            lefthand_in_fov = self.in_fov(position_lefthand_head, self.fov_h, self.fov_v)   # self.fov_h, self.fov_v
            righthand_in_fov = self.in_fov(position_righthand_head, self.fov_h, self.fov_v)            

            if self.full_hand_visibility:
                lefthand_in_fov = torch.ones(lefthand_in_fov.shape, dtype=torch.bool)   # disable fov
                righthand_in_fov = torch.ones(righthand_in_fov.shape, dtype=torch.bool)  # disable fov


            return {'sparse': input_hmd[::self.sample_stride],
                    'poses_gt': output_gt[::self.sample_stride],
#                    'betas_gt': np.repeat(betas[::self.sample_stride][np.newaxis], repeats=self.window_size, axis=0),
                    'fov_l':lefthand_in_fov[::self.sample_stride],
                    'fov_r':  righthand_in_fov[::self.sample_stride],
                    'head_trans4x4_global':head_global_trans_list[[frame + self.window_size -1],...],
                    'root_trans':root_trans[[frame + self.window_size - 1],...],
                    'offset': self.offset,
                    'filename': filename
                    }

        else:

            input_hmd  = hmd_position_global_full_gt_list.reshape(hmd_position_global_full_gt_list.shape[0], -1)[1:].float()

            input_hmd[:,36:38] = input_hmd[:,36:38] + self.offset
            input_hmd[:,39:41] = input_hmd[:,39:41] + self.offset
            input_hmd[:,42:44] = input_hmd[:,42:44] + self.offset
            head_global_trans_list = head_global_trans_list[1:]
            head_global_trans_list[:,:2,-1] = head_global_trans_list[:,:2,-1] + self.offset
            output_gt = rotation_local_full_gt_list[1:].float()

            head_global_trans_inv_list = torch.inverse(head_global_trans_list)
            position_lefthand_gt_world = input_hmd[:,36+3:36+6]
            position_lefthand_gt_world_aug = torch.cat([position_lefthand_gt_world, torch.ones(position_lefthand_gt_world.shape[0],1)],dim=1)

            position_lefthand_head = torch.matmul(head_global_trans_inv_list,position_lefthand_gt_world_aug.unsqueeze(-1)).squeeze()[:,:3]
            position_righthand_gt_world = input_hmd[:,36+6:36+9]
            position_righthand_gt_world_aug = torch.cat([position_righthand_gt_world, torch.ones(position_righthand_gt_world.shape[0],1)],dim=1)
            position_righthand_head = torch.matmul(head_global_trans_inv_list,position_righthand_gt_world_aug.unsqueeze(-1)).squeeze()[:,:3]

            lefthand_in_fov = self.in_fov(position_lefthand_head, self.fov_h, self.fov_v)
            righthand_in_fov = self.in_fov(position_righthand_head, self.fov_h, self.fov_v)

            if self.full_hand_visibility:
                lefthand_in_fov = torch.ones(lefthand_in_fov.shape, dtype=torch.bool)   
                righthand_in_fov = torch.ones(righthand_in_fov.shape, dtype=torch.bool) 


            return {'sparse': input_hmd,
                    'poses_gt': output_gt,
#                    'betas_gt': np.repeat(betas[np.newaxis], repeats=input_hmd.shape[0], axis=0),
                    'fov_l':lefthand_in_fov,
                    'fov_r':  righthand_in_fov,
                    'head_trans4x4_global':head_global_trans_list[0:],
                    'root_trans':root_trans[2:],
                    'offset': self.offset,
                    'filename': filename
                    }
