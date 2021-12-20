"""
 Copyright (C) 2021-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import torch
import torch.nn as nn
from mbv3 import mobilenet_v3_small, InvertedResidual
import math
import copy
import torch.nn.functional as F

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out
    
class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2**i, num_f_maps, num_f_maps) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
fusion_ops = ['max', 'concat']

class Late_Fusion_Classifier(nn.Module):
    def __init__(self, n_class, in_features, n_view, fusion_op='max', use_refinement_layers=False):
        super(Late_Fusion_Classifier, self).__init__()
        self.fusion_op = fusion_op
        self.use_refinement_layers = use_refinement_layers
        self.linear_0 = nn.Linear(in_features=in_features, out_features=in_features//2)
        self.linear_1 = nn.Linear(in_features=self.linear_0.out_features, out_features=n_class)
        if fusion_op == 'concat':
            self.linear_fc = nn.Linear(in_features=in_features*n_view, out_features=in_features)
        if use_refinement_layers:
            self.refinement_layers = nn.ModuleList([Refinement(num_layers=4, num_f_maps=n_class*2, dim=n_class, num_classes=n_class) for s in range(2)])
    
    def max_fusion(self, x1, x2):
        x = torch.stack([x1, x2])
        x = torch.max(x, dim=0).values
        
        x = self.linear_0(x)
        x = torch.relu(x)
        
        x = self.linear_1(x)
#         x = torch.softmax(x, dim=1)
        return x
    
    def concat_fusion(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        x = self.linear_fc(x)
        x = torch.relu(x)
        
        x = self.linear_0(x)
        x = torch.relu(x)
        
        x = self.linear_1(x)
        x[:,0] = torch.sigmoid(x[:, 0])
        return x
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception as e:
                print('Unmatched weight at {}: {}'.format(name, e))
                
    def forward(self, x1, x2):
        '''
        input: x is a list of input tensors [x1, x2, ...] representing output feature vectors from different views.
        output: N class prediction logits.
        '''
        if self.fusion_op == 'max':
            x = self.max_fusion(x1, x2)
        elif self.fusion_op == 'concat':
            x = self.concat_fusion(x1, x2)
        else:
            assert self.fusion_op in ['max', 'concat'], 'Operation not supported. Please select one of {}'.format(fusion_ops)
            
        if self.use_refinement_layers:
            outputs = []

            for R in self.refinement_layers:
                x = torch.unsqueeze(x, -1)
                x = R(x)
                x = torch.squeeze(x, -1)
                outputs += [x]
            return outputs
        else:
            return x

def create_mbv3s_model(img_channel, n_class, n_view, fusion_op, use_refinement_layers=False):
    
    backbone = mobilenet_v3_small(pretrained=True)

    in_features = backbone.classifier[0].in_features
    backbone.classifier = nn.Identity()
    
    
    classifier = Late_Fusion_Classifier(n_class=n_class, in_features=in_features, n_view=n_view, fusion_op=fusion_op, use_refinement_layers=use_refinement_layers)

    return backbone, classifier


def reset_shifter(net):
    for block in net.features:
        if isinstance(block, InvertedResidual):
            block.shift.reset_module()
            pass
