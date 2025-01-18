'''
# --------------------------------------------
# code for model optimization and testing, adapted from AvatarPoser
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from models.select_model import define_net, define_bm
from models.model_base import ModelBase
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils import utils_transform
from IPython import embed

class ModelEgoPoser(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelEgoPoser, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.net = define_net(opt)
        self.net = self.model_to_device(self.net)
        self.window_size = self.opt['datasets']['test']['window_size']
#        self.bm = self.net.module.body_model

        self.opt_bm = self.opt['body_model']
        self.bm_dict = define_bm(opt)
        default_gender = self.opt_bm['default_gender']
        self.bm = self.bm_dict[default_gender]
    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.net.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def init_test(self):
        self.load(test=True)                           # load model
        self.log_dict = OrderedDict()         # log
    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self, test=False):
        load_path_net = self.opt['path']['pretrained_net'] if test == False else self.opt['path']['pretrained']
        if load_path_net is not None:
            print('Loading model for motion tracking [{:s}] ...'.format(load_path_net))
            self.load_network(load_path_net, self.net, strict=self.opt_train['param_strict'], param_key='params')

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizer = self.opt['path']['pretrained_optimizer']
        if load_path_optimizer is not None and self.opt_train['optimizer_reuse']:
            print('Loading optimizer [{:s}] ...'.format(load_path_optimizer))
            self.load_optimizer(load_path_optimizer, self.optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.net, iter_label)
        if self.opt_train['optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.optimizer, 'optimizer', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        lossfn_type = self.opt_train['lossfn_type']
        if lossfn_type == 'l1':
            self.lossfn = nn.L1Loss().to(self.device)
        elif lossfn_type == 'l2':
            self.lossfn = nn.MSELoss().to(self.device)
        elif lossfn_type == 'l2sum':
            self.lossfn = nn.MSELoss(reduction='sum').to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(lossfn_type))
        self.global_orientation_weight = self.opt_train['global_orientation_weight']
        self.joint_rotation_weight = self.opt_train['joint_rotation_weight']
        self.joint_position_weight = self.opt_train['joint_position_weight']
        self.shape_weight = self.opt_train['shape_weight']
    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.optimizer = Adam(optim_params, lr=self.opt_train['optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer,
                                                        self.opt_train['scheduler_milestones'],
                                                        self.opt_train['scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):

        self.sparse = data['sparse'].to(self.device)
        self.head_trans4x4_global = data['head_trans4x4_global'].to(self.device)
        self.GT_global_orientation = data['poses_gt'][...,:6].to(self.device)
        self.GT_joint_rotation = data['poses_gt'][...,6:].to(self.device)
        self.GT_pos_root = data['root_trans'].to(self.device)
        self.GT_betas = data['betas_gt'].to(self.device) if 'betas_gt' in data else None
        self.GT_joint_position = self.fk(body_model=self.bm, global_orientation=self.GT_global_orientation, joint_rotation=self.GT_joint_rotation, betas=self.GT_betas)

        self.fov_l = data['fov_l']
        self.fov_r = data['fov_r']
        self.offset = data['offset'].to(self.device)
    # ----------------------------------------
    # feed L to net
    # ----------------------------------------
    def net_forward(self):

        x = {'sparse_input':self.sparse, 'fov_l':self.fov_l, 'fov_r':self.fov_r}
        output = self.net(x)
        self.E_global_orientation = output['root_orient']
        self.E_joint_rotation = output['pose_body']
        self.E_betas = output['betas']
        self.E_joint_position = self.fk(body_model=self.bm, global_orientation=self.E_global_orientation, joint_rotation=self.E_joint_rotation, betas=self.E_betas)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.optimizer.zero_grad()
        self.net_forward()

        joint_rotation_loss = self.lossfn(self.E_joint_rotation, self.GT_joint_rotation[:,-1,...])
        global_orientation_loss = self.lossfn(self.E_global_orientation, self.GT_global_orientation[:,-1,...])
        joint_position_loss = self.lossfn(self.E_joint_position, (self.GT_joint_position[:,-1,...])) 
        shape_regularization = self.lossfn(self.E_betas, torch.zeros_like(self.E_betas)) if self.E_betas is not None else torch.tensor(0).to(self.device)
        shape_loss = self.lossfn(self.E_betas, self.GT_betas[:,-1,...]) if self.E_betas is not None else torch.tensor(0).to(self.device)

        loss =  self.global_orientation_weight *global_orientation_loss \
              + self.joint_rotation_weight * joint_rotation_loss  \
              + self.joint_position_weight *joint_position_loss \
              + self.shape_weight * shape_regularization

        loss.backward()
        self.optimizer.step()


        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        optimizer_clipgrad = self.opt_train['optimizer_clipgrad'] if self.opt_train['optimizer_clipgrad'] else 0
        if optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.net.module.parameters(), max_norm=self.opt_train['optimizer_clipgrad'], norm_type=2)


        # ------------------------------------
        # regularizer
        # ------------------------------------
        regularizer_orthstep = self.opt_train['regularizer_orthstep'] if self.opt_train['regularizer_orthstep'] else 0
        if regularizer_orthstep > 0 and current_step % regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.net.apply(regularizer_orth)
        regularizer_clipstep = self.opt_train['regularizer_clipstep'] if self.opt_train['regularizer_clipstep'] else 0
        if regularizer_clipstep > 0 and current_step % regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.net.apply(regularizer_clip)

        self.log_dict['total_loss'] = loss.item()
        self.log_dict['global_orientation_loss'] = global_orientation_loss.item()
        self.log_dict['joint_rotation_loss'] = joint_rotation_loss.item()
        self.log_dict['joint_position_loss'] = joint_position_loss.item()
        self.log_dict['shape_loss'] = shape_loss.item()


    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):         # input length = window_size, output length = 1

            self.net.eval()

            self.sparse = self.sparse.squeeze()
      
            self.head_trans4x4_global = self.head_trans4x4_global.squeeze()
            test_batch = self.opt['datasets']['test']['test_batch']

            with torch.no_grad():


                num_frames = self.sparse.shape[0]
                num_batch = (num_frames-self.window_size)//test_batch+1
                init_frame = self.window_size-1

                E_global_orientation_list = []  
                E_joint_rotation_list = []      

                for idx_batch in range(num_batch):
                    self.sparse_segment = self.sparse[test_batch *idx_batch:test_batch * (idx_batch+1)+self.window_size-1]
                    self.fov_l_segment = self.fov_l[...,test_batch *idx_batch:test_batch * (idx_batch+1)+self.window_size-1]
                    self.fov_r_segment = self.fov_r[...,test_batch *idx_batch:test_batch * (idx_batch+1)+self.window_size-1]

                    input_list_current = []
                    fov_l_list_current = []
                    fov_r_list_current = []
                    
                    for frame_idx in range(init_frame, self.sparse_segment.shape[0]):
                        input_list_current.append(self.sparse_segment[frame_idx-init_frame:frame_idx+1,...].unsqueeze(0))
                        fov_l_list_current.append(self.fov_l_segment[...,frame_idx-init_frame:frame_idx+1])
                        fov_r_list_current.append(self.fov_r_segment[...,frame_idx-init_frame:frame_idx+1])

                    input_tensor_current = torch.cat(input_list_current, dim = 0)
                    fov_l_tensor_current = torch.cat(fov_l_list_current, dim = 0)
                    fov_r_tensor_current = torch.cat(fov_r_list_current, dim = 0)


                    x = {'sparse_input': input_tensor_current,
                         'fov_l': fov_l_tensor_current,
                         'fov_r': fov_r_tensor_current}

                    output = self.net(x)
                    E_global_orientation_current = output['root_orient']
                    E_joint_rotation_current = output['pose_body']

                    E_global_orientation_list.append(E_global_orientation_current)
                    E_joint_rotation_list.append(E_joint_rotation_current)
                
                E_global_orientation_tensor = torch.cat(E_global_orientation_list, dim=0)
                E_joint_rotation_tensor = torch.cat(E_joint_rotation_list, dim=0)

            self.E = torch.cat([E_global_orientation_tensor, E_joint_rotation_tensor],dim=-1).to(self.device).squeeze()
            predicted_angle = utils_transform.sixd2aa(self.E[:,:132].reshape(-1,6).detach()).reshape(self.E[:,:132].shape[0],-1).float()
            t_head2world = self.head_trans4x4_global[:,:3,3][init_frame:]
            body_pose_local=self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3]})
            t_head2root = body_pose_local.Jtr[:,15,:]
            t_root2world = -t_head2root+t_head2world
            self.predicted_translation = t_root2world
            self.predicted_body=self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': self.predicted_translation}) 


            self.predicted_position = self.predicted_body.Jtr[:,:22,:]
            self.predicted_angle = predicted_angle


            self.net.train()




    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_prediction(self,):
        body_parms = OrderedDict()
        body_parms['pose_body'] = self.predicted_angle[...,3:66]
        body_parms['root_orient'] = self.predicted_angle[...,:3]
        body_parms['trans'] = self.predicted_translation
        body_parms['position'] = self.predicted_position       
        body_parms['body'] = self.predicted_body

        return body_parms

    def current_gt(self, ):
        init_frame = self.window_size -1
        num_frames = self.GT_joint_rotation.squeeze().shape[0]
        body_parms = OrderedDict()
        body_parms['pose_body'] = utils_transform.sixd2aa(self.GT_joint_rotation.squeeze().reshape(num_frames,-1,6),batch=True).reshape(num_frames,-1)[init_frame:]
        body_parms['root_orient'] = utils_transform.sixd2aa(self.GT_global_orientation.squeeze())[init_frame:] 
        body_parms['trans'] = self.GT_pos_root.squeeze()[init_frame:]
        joint_position = self.GT_joint_position + self.GT_pos_root.squeeze().unsqueeze(1)
        joint_position[...,:2]+=self.offset
        body_parms['position'] = joint_position[init_frame:]
        body_parms['body'] = self.bm(**{k:v for k,v in body_parms.items() if k in  ['pose_body', 'trans', 'root_orient']})
        body_parms['fov_l'] = self.fov_l[:,init_frame:]
        body_parms['fov_r'] = self.fov_r[:,init_frame:]
        return body_parms


    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.net)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.net)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.net)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.net)
        return msg

    # ----------------------------------------
    # forward kinematics
    # ----------------------------------------

    def fk(self, body_model, global_orientation, joint_rotation, trans=None, betas=None):

        bs = global_orientation.shape[0]
        len_seq = global_orientation.shape[1] if len(global_orientation.shape) == 3 else 1
        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1,6)).reshape(-1,3).float()
        joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1,6)).reshape(-1,63).float()

        if betas is not None:
            betas = betas.reshape(-1,16).float()
        body_pose = body_model(**{'pose_body':joint_rotation, 'root_orient':global_orientation, 'trans': trans, 'betas':betas})
        joint_position = body_pose.Jtr[...,:22,:].reshape(bs,len_seq,-1,3).squeeze()
        return joint_position
