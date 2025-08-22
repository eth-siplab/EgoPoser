'''
# --------------------------------------------
# EgoPoser neural network
# --------------------------------------------
# EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere (ECCV 2024)
# https://github.com/eth-siplab/EgoPoser
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import torch
import torch.nn as nn

class AvatarNet(nn.Module):
    def __init__(self, input_dim, num_layer, embed_dim, nhead, spatial_normalization, shape_estimation=False):
        super(AvatarNet, self).__init__()

        self.linear_embedding = nn.Linear(input_dim,embed_dim)
        self.linear_embedding_fast = nn.Linear(input_dim,embed_dim//2)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        

        self.global_orientation_decoder = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.ReLU(),
                            nn.Linear(embed_dim, 6)
            )
        self.joint_rotation_decoder = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.ReLU(),
                            nn.Linear(embed_dim, 126)
            )
        
        self.shape_decoder = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.ReLU(),
                            nn.Linear(embed_dim, 16)
            ) if shape_estimation else None
        
        self.spatial_normalization = spatial_normalization


    def input_masking(self, input_tensor, fov_l, fov_r):

        lefthand_idx = [*range(6,12),*range(24,30),*range(39,42),*range(48,51)]
        righthand_idx = [*range(12,18),*range(30,36),*range(42,45),*range(51,54)]

        lefthand_out_fov_selector=torch.zeros(input_tensor.shape, dtype=torch.bool) 
        lefthand_out_fov_selector[fov_l==False]=True
        no_lefthand_selector=torch.ones(input_tensor.shape[2], dtype=torch.bool)
        no_lefthand_selector[lefthand_idx]=False
        lefthand_out_fov_selector[...,no_lefthand_selector] = False
        input_tensor[lefthand_out_fov_selector]=0

        righthand_out_fov_selector=torch.zeros(input_tensor.shape, dtype=torch.bool) 
        righthand_out_fov_selector[fov_r==False]=True
        no_righthand_selector=torch.ones(input_tensor.shape[2], dtype=torch.bool)
        no_righthand_selector[righthand_idx]=False
        righthand_out_fov_selector[...,no_righthand_selector] = False
        input_tensor[righthand_out_fov_selector]=0

        return input_tensor

    def forward(self, x):
        input_tensor = x['sparse_input']
        fov_l = x['fov_l']
        fov_r = x['fov_r']


        if self.spatial_normalization:
            # spatial normalization, make horizontal positions relative to the head while keeping the global vertial positions
            head_horizontal_trans = input_tensor.clone()[...,36:38].detach()
            input_tensor[...,36:38] -= head_horizontal_trans
            input_tensor[...,39:41] -= head_horizontal_trans
            input_tensor[...,42:44] -= head_horizontal_trans

        # temporal normalization, make all frames within a window relative to the first frame

        delta_0 = input_tensor[...,36:38] - input_tensor[...,[0],36:38]
        delta_1 = input_tensor[...,39:41] - input_tensor[...,[0],39:41]
        delta_2 = input_tensor[...,42:44] - input_tensor[...,[0],42:44]

        input_tensor = torch.cat([input_tensor, delta_0, delta_1, delta_2],dim=-1)

        # field of view masking

        input_tensor = self.input_masking(input_tensor, fov_l, fov_r)

        # SlowFast fusion
        x_fast = input_tensor[:,-input_tensor.shape[1]//2:,...]
        x_slow = input_tensor[:,::2,...]

        x_fast = self.linear_embedding(x_fast)
        x_slow = self.linear_embedding(x_slow)
        x = x_fast + x_slow

        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)
        x = x[:, -1]

        root_orient = self.global_orientation_decoder(x)
        pose_body = self.joint_rotation_decoder(x) 
        betas = self.shape_decoder(x) if self.shape_decoder is not None else None

        output = {}
        output['root_orient'] = root_orient
        output['pose_body'] = pose_body
        output['betas'] = betas
        return output