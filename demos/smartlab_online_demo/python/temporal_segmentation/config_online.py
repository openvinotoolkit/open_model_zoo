# -*- coding: utf-8 -*-

import os.path as osp
from easydict import EasyDict

__C = EasyDict()
current_dir = osp.dirname(__file__)

__C.img_size_h = 224
__C.img_size_w = 224
# ************************** Hyper-params For Feature Embedding (I3D) **************************
__C.embed_window_length = 16  # 视频特征提取的长度单元
__C.embed_batch_size = 1
__C.embed_dim = 1024

__C.embed_model_dir = {  # set the path where to load pretrained i3d models
    'rgb': osp.join(current_dir, 'i3d_model_dir/rgb_scratch/model.ckpt'),
    'rgb600': osp.join(current_dir, 'i3d_model_dir/rgb_scratch_kin600/model.ckpt'),
    'flow': osp.join(current_dir, 'i3d_model_dir/flow_scratch/model.ckpt'),
    'rgb_imagenet': osp.join(current_dir, 'i3d_model_dir/rgb_imagenet/model.ckpt'),
    'flow_imagenet': osp.join(current_dir, 'i3d_model_dir/flow_imagenet/model.ckpt'),
}

# ************************** Hyper-params For Action Segmentation (MSTCN) **************************
__C.seg_window_length = 16  # 视频分割的长度单元
__C.seg_window_step = 10  # 视频分割的滑动步长

__C.dataset = "tianping"
__C.seg_model_dir = osp.join(current_dir, "mstcn_model_dir", __C.dataset)
__C.mapping_file = osp.join(__C.seg_model_dir, "mapping.txt")
with open(__C.mapping_file, 'r') as f:
    mapping_info = f.readlines()
mapping_info = [i.split() for i in mapping_info]
__C.mapping_dict = dict(zip([i[0] for i in mapping_info], [i[-1] for i in mapping_info]))
__C.num_classes = len(__C.mapping_dict)

__C.num_layers_PG = 11
__C.num_layers_R = 10
__C.num_R = 3
__C.num_f_maps = 64

online_opt = __C
