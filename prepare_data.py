'''
# --------------------------------------------
# data preprocessing for AMASS dataset
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import torch
import numpy as np
import os
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose
from utils import utils_transform
import pickle

dataroot_amass ="../../Datasets/amass" # root of amass dataset

for dataroot_subset in ["MPI_HDM05", "BioMotionLab_NTroje", "CMU"]:
    print(dataroot_subset)
    for phase in ["train","test"]:
        print(phase)

        split_file = os.path.join("./data_split", dataroot_subset, phase+"_split.txt")

        with open(split_file, 'r') as f:
            filepaths = [line.rstrip('\n') for line in f]


        rotation_local_full_gt_list = []

        hmd_position_global_full_gt_list = []

        body_parms_list = []

        head_global_trans_list = []


        support_dir = 'support_data/'
        bm_fname_male = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('male'))
        dmpl_fname_male = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('male'))

        bm_fname_female = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('female'))
        dmpl_fname_female = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('female'))

        bm_fname_neutral = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('neutral'))
        dmpl_fname_neutral = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('neutral'))

        num_betas = 16 # number of body parameters
        num_dmpls = 8 # number of DMPL parameters
        bm_male = BodyModel(bm_fname=bm_fname_male, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_male)#.to(comp_device)
        bm_female = BodyModel(bm_fname=bm_fname_female, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_female)
        bm_neutral = BodyModel(bm_fname=bm_fname_neutral, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_neutral)

        idx = 0
        for filepath in filepaths:
            data = dict()
            filepath_ = os.path.join(dataroot_amass, filepath)
            bdata = np.load(filepath_,allow_pickle=True)
            try:
                framerate = bdata["mocap_framerate"]
                print("framerate is {}".format(framerate))
            except:
                print(filepath)
                print(list(bdata.keys()))       
                continue                          # skip shape.npz

            idx+=1
            print(idx)

            if framerate == 120:
                stride = 2
            elif framerate == 60:
                stride = 1

            bdata_poses = bdata["poses"][::stride,...]
            bdata_trans = bdata["trans"][::stride,...]
            bdata_betas = bdata["betas"] 
            subject_gender = bdata["gender"] 

            bm = bm_male # bm_male # if subject_gender == 'male' else bm_female


            body_parms = {
                'root_orient': torch.Tensor(bdata_poses[:, :3]),#.to(comp_device), # controls the global root orientation
                'pose_body': torch.Tensor(bdata_poses[:, 3:66]),#.to(comp_device), # controls the body
                'trans': torch.Tensor(bdata_trans),#.to(comp_device), # controls the global body position
            }

            body_parms_list = body_parms

            body_pose_world=bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','root_orient','trans']})

            output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1,3)
            output_6d = utils_transform.aa2sixd(output_aa).reshape(bdata_poses.shape[0],-1)
            rotation_local_full_gt_list = output_6d[1:]

            rotation_local_matrot = aa2matrot(torch.tensor(bdata_poses).reshape(-1,3)).reshape(bdata_poses.shape[0],-1,9)
            rotation_global_matrot = local2global_pose(rotation_local_matrot, bm.kintree_table[0].long()) # rotation of joints relative to the origin

            head_rotation_global_matrot = rotation_global_matrot[:,[15],:,:]

            rotation_global_6d = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1,3,3)).reshape(rotation_global_matrot.shape[0],-1,6)
            input_rotation_global_6d = rotation_global_6d[1:,[15,20,21],:]

            rotation_velocity_global_matrot = torch.matmul(torch.inverse(rotation_global_matrot[:-1]),rotation_global_matrot[1:])
            rotation_velocity_global_6d = utils_transform.matrot2sixd(rotation_velocity_global_matrot.reshape(-1,3,3)).reshape(rotation_velocity_global_matrot.shape[0],-1,6)
            input_rotation_velocity_global_6d = rotation_velocity_global_6d[:,[15,20,21],:]

            position_global_full_gt_world = body_pose_world.Jtr[:,:22,:] # position of joints relative to the world origin

            position_head_world = position_global_full_gt_world[:,15,:] # world position of head

            head_global_trans = torch.eye(4).repeat(position_head_world.shape[0],1,1)
            head_global_trans[:,:3,:3] = head_rotation_global_matrot.squeeze()
            head_global_trans[:,:3,3] = position_global_full_gt_world[:,15,:]

            head_global_trans_list = head_global_trans[1:]


            num_frames = position_global_full_gt_world.shape[0]-1


            hmd_position_global_full_gt_list = torch.cat([
                                                                    input_rotation_global_6d.reshape(num_frames,-1), #0-18 (h0-6,l6-12,r12-18)
                                                                    input_rotation_velocity_global_6d.reshape(num_frames,-1), #18-36(h18-24,l24-30,r30-36)
                                                                    position_global_full_gt_world[1:, [15,20,21], :].reshape(num_frames,-1), #36-45(h36-39,l39-42,r42-45)
                                                                    position_global_full_gt_world[1:, [15,20,21], :].reshape(num_frames,-1)- #45-54(h45-48,l48-51,r51-54)
                                                                    position_global_full_gt_world[:-1, [15,20,21], :].reshape(num_frames,-1)], dim=-1)


            print(str(idx)+'/'+str(len(filepaths)))

            data['poses'] = bdata_poses
            data['trans'] = bdata_trans
            data['betas'] = bdata_betas
            data['framerate'] = 60
            data['gender'] = subject_gender

            data['rotation_local_full_gt_list'] = rotation_local_full_gt_list
            data['hmd_position_global_full_gt_list'] = hmd_position_global_full_gt_list
            data['body_parms_list'] = body_parms_list
            data['head_global_trans_list'] = head_global_trans_list

            subject_name = filepath.split('/')[-2]
            savedir = os.path.join("./data_amass_avatarposer_split", phase, dataroot_subset, subject_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            filename = filepath.split('/')[-1][:-4]

            with open(os.path.join(savedir,'{}.pkl'.format(filename)), 'wb') as f:
                pickle.dump(data, f)
