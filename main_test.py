'''
# --------------------------------------------
# EgoPoser main testing
# --------------------------------------------
# EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere (ECCV 2024)
# https://github.com/eth-siplab/EgoPoser
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import os.path
import argparse
import random
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from tqdm import tqdm


def main(yaml_path='options/test_egoposer.yaml'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=yaml_path, help='Path to option YAML file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)


    opt['path']['pretrained'] = opt['pretrained_model']


    current_step = 0

    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    logger_name = 'test'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    dataset_opt = opt['datasets']['test']

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                shuffle=False, num_workers=1,
                                drop_last=False, pin_memory=True)

    model = define_Model(opt)

    model.load(test=True)

    logger.info(model.info_network())
    logger.info(model.info_params())


    test_small = opt['test_small']
    if test_small:
        test_keywords = opt['test_keywords']


    rot_error = []
    pos_error = []
    vel_error = []

    for index, test_data in enumerate(tqdm(test_loader)):

        fname = test_data['filename']

        if test_small:
            if test_keywords not in fname[0]:
                continue

        model.feed_data(test_data)
        frame_length = model.sparse.shape[1]

        if frame_length<=80:
            continue

        model.test()
        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()
        predicted_angle = body_parms_pred['pose_body']
        predicted_position = body_parms_pred['position']

        gt_angle = body_parms_gt['pose_body']
        gt_position = body_parms_gt['position']

        save_trans = body_parms_pred['trans'].cpu().numpy()
        save_poses = torch.cat([body_parms_pred['root_orient'],body_parms_pred['pose_body']],dim=1).cpu().numpy()
        save_smpl_dir = os.path.join(opt['path']['smpl_pred'],str(current_step), fname[0].split('/')[-3], fname[0].split('/')[-2])

        if not os.path.exists(save_smpl_dir):
            os.makedirs(save_smpl_dir)
        filename = fname[0].split('/')[-1][:-4]
        save_smpl_parms_path = os.path.join(save_smpl_dir,'{}.npz'.format(filename))
        np.savez(save_smpl_parms_path, trans=save_trans, poses=save_poses)


        save_trans_gt = body_parms_gt['trans'].cpu().numpy()
        save_poses_gt = torch.cat([body_parms_gt['root_orient'],body_parms_gt['pose_body']],dim=1).cpu().numpy()
        save_smpl_gt_dir = os.path.join(opt['path']['smpl_pred'],'gt', fname[0].split('/')[-3], fname[0].split('/')[-2])

        if not os.path.exists(save_smpl_gt_dir):
            os.makedirs(save_smpl_gt_dir)
        filename = fname[0].split('/')[-1][:-4]
        save_smpl_parms_gt_path = os.path.join(save_smpl_gt_dir,'{}.npz'.format(filename))
        np.savez(save_smpl_parms_gt_path, trans=save_trans_gt, poses=save_poses_gt)


        predicted_angle = predicted_angle.reshape(body_parms_pred['pose_body'].shape[0],-1,3)                    
        gt_angle = gt_angle.reshape(body_parms_gt['pose_body'].shape[0],-1,3)

        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*60
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*60

        rot_error_ = torch.mean(torch.absolute(gt_angle-predicted_angle))
        pos_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1)))
        vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))

        rot_error.append(rot_error_)
        pos_error.append(pos_error_)
        vel_error.append(vel_error_)

        if opt['print_all']:
            logger.info("testing the sample {}/{}".format(index, len(test_loader)))
            logger.info('Result of file {}:'.format(filename))
            logger.info('{} rotation error: {:<.5f}'.format(fname, rot_error_*57.2958))
            logger.info('{} positional error: {:<.5f}'.format(fname, pos_error_*100))
            logger.info('{} velocity error: {:<.5f}'.format(fname, vel_error_*100))


    rot_error = sum(rot_error)/len(rot_error)
    pos_error = sum(pos_error)/len(pos_error)
    vel_error = sum(vel_error)/len(vel_error)


    # testing log
    logger.info('Average rotational error [degree]: {:<.3f}, Average positional error [cm]: {:<.3f}, Average velocity error [cm/s]: {:<.3f} \n'.format(rot_error*57.2958, pos_error*100, vel_error*100))


if __name__ == '__main__':
    main()
