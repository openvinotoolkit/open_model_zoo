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
from tabulate import tabulate
from openvino.inference_engine import IECore

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
class MwGlobalExp:
    def __init__(self,
        num_classes,
        fp_model,
        root_input,
        conf_thresh=0.2,
        nms_thresh=0.3,
        is_show=False
    ):
        self.print_interval = 100
        self.eval_interval = 10
        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.00
        self.width = 1.00
        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = None
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65

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
        if fp_model.endswith('.bin'):
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

    def get_openvino_model(self, device='CPU'):
        assert self.fp_model.endswith('.bin')
        ie = IECore()
        net = ie.read_network(
            model=self.fp_model[:-4]+".xml",
            weights=self.fp_model[:-4]+".bin"
        )

        input_name = next(iter(net.input_info))
        output_name = next(iter(net.outputs))
        net.input_info[input_name].precision = 'FP32'
        _, _, h, w = net.input_info[input_name].input_data.shape
        net.outputs[output_name].precision = 'FP16'

        return (
            input_name,
            output_name,
            (h, w), 
            ie.load_network(network=net, device_name=device)
            )

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")
