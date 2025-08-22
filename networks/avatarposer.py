'''
# --------------------------------------------
# AvatarPoser neural network
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (jiaxi.jiang@inf.ethz.ch)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import torch.nn as nn


    # ----------------------------------------
    # Input dimension: B*40*54
    # Output dimension: B*1*(126+6)
    # ----------------------------------------


class AvatarNet(nn.Module):
    def __init__(self, input_dim, num_layer, embed_dim, nhead, spatial_normalization):
        super(AvatarNet, self).__init__()

        self.linear_embedding = nn.Linear(input_dim,embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        


        self.stabilizer = nn.Sequential(
                            nn.Linear(embed_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 6)
            )
        self.joint_rotation_decoder = nn.Sequential(
                            nn.Linear(embed_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 126)
            )
        self.spatial_normalization = spatial_normalization

    def forward(self, x):

        input_tensor = x['sparse_input']

        if self.spatial_normalization:
            # spatial normalization, make horizontal positions relative to the head
            
            head_horizontal_trans = input_tensor.clone()[...,36:38].detach()
            input_tensor[...,36:38] -= head_horizontal_trans
            input_tensor[...,39:41] -= head_horizontal_trans
            input_tensor[...,42:44] -= head_horizontal_trans            
            
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)
        x = x[:, -1]
        root_orient = self.stabilizer(x)
        pose_body = self.joint_rotation_decoder(x) 
        output = {}
        output['root_orient'] = root_orient
        output['pose_body'] = pose_body
        return output