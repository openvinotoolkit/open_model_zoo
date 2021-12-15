# -*- coding: utf-8 -*-

from pathlib import Path
from easydict import EasyDict

current_dir = Path(__file__).parent.absolute()

__C = EasyDict()

__C.img_size_h = 224
__C.img_size_w = 224
__C.sliding_smoothing = False
# ************************** Hyper-params For Feature Embedding (I3D) **************************
__C.embed_window_length = 16  # 视频特征提取的长度单元
__C.embed_window_stride = 1  # 视频特征提取的窗口滑动步长
__C.embed_window_atrous_rate = 3  # 隔帧采样
__C.embed_batch_size = 1
__C.embed_dim = 1024

# set the path where to load pretrained i3d models
__C.embed_model_dir = {'rgb': Path(current_dir.parent, 'model_dir/embed/i3d/rgb_scratch/model.ckpt'),
                       'rgb600': Path(current_dir.parent, 'model_dir/embed/i3d/rgb_scratch_kin600/model.ckpt'),
                       'flow': Path(current_dir.parent, 'model_dir/embed/i3d/flow_scratch/model.ckpt'),
                       'rgb_imagenet': Path(current_dir.parent, 'model_dir/embed/i3d/rgb_imagenet/model.ckpt'),
                       'flow_imagenet': Path(current_dir.parent, 'model_dir/embed/i3d/flow_imagenet/model.ckpt'), }

# ************************** Hyper-params For Action Segmentation (MSTCN) **************************
__C.num_layers_PG = 11
__C.num_layers_R = 10
__C.num_R = 3
__C.num_f_maps = 64
__C.seg_model_dir = Path(current_dir.parent, 'model_dir/seg/mstcn_online').absolute()
mapping_file = Path(__C.seg_model_dir, "mapping.txt")
with open(mapping_file, 'r') as f:
    mapping_info = f.readlines()
mapping_info = [i.split() for i in mapping_info]
__C.mapping_dict = dict(zip([i[0] for i in mapping_info], [i[-1] for i in mapping_info]))
__C.num_classes = len(__C.mapping_dict)

__C.seg_batch_size = 24  # the maxmum process unit of MSTCN (24 embeddings)

config = __C
