#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
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

import os
import torch.nn as nn
import sys
sys.path.append('object_detection')
from tools.yolox_exp import Exp as MyExp
from model.yolox import YOLOX, YOLOPAFPN, YOLOXHead


# class maps
mw_glb2acls6 = (
  "balance", 
  "box", 
  "tray", 
  "ruler", 
  "hand", 
  "scale"
)

mw_glb2bcls3 = (
  'weights',
  'tweezers',
  'battery'
)

mw_glb1cls10 = (
  "balance",
  "weights",
  "tweezers",
  "box",
  "battery",
  "tray",
  "ruler",
  "rider",
  "scale",
  "hand"
)


# global setting of obj-det
class MwGlobalExp(MyExp):
    def __init__(self, num_classes, fp_model, root_input, conf_thresh=0.2, nms_thresh=0.3, is_show=False):
      super(MwGlobalExp, self).__init__()
      self.depth = 0.33
      self.width = 0.25
      self.input_size = (416, 416)
      self.mosaic_scale = (0.5, 1.5)
      self.random_size = (10, 20)
      self.test_size = (416, 416)
      self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
      self.enable_mixup = False
      # Define yourself dataset path
      self.data_dir = ""
      self.train_ann = ""
      self.val_ann = ""
      # define class map
      self.num_classes = num_classes
      if self.num_classes == 10:
        self.mw_classes = mw_glb1cls10
      elif self.num_classes == 6:
        self.mw_classes = mw_glb2acls6
      elif self.num_classes == 3:
        self.mw_classes = mw_glb2bcls3
      else:
        raise ValueError
      # define model file
      if fp_model.endswith('.pth') \
        or fp_model.endswith('.onnx') \
        or fp_model.endswith('.xml'):
        self.fp_model = fp_model
      # define dataset
      if root_input is not None:
        self.root_imgs = root_input
        support_suffices = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
        other_suffices = [s.upper() for s in support_suffices]
        support_suffices += other_suffices
        self.img_suffix = None
        for f in os.listdir(self.root_imgs):
          suffix = '.'+f.split('.')[-1]
          if suffix in support_suffices:
            self.img_suffix = suffix
            break
        assert self.img_suffix is not None
      # define preview method
      self.is_show = is_show
      # define conditions
      self.confthre = conf_thresh
      self.nmsthre = nms_thresh

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

