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
    def __init__(self, num_classes, root_input, fp_model, nms_thresh, conf_thresh, ie, device):
        if num_classes == 10:
            self.mw_classes = mw_glb1cls10
        elif num_classes == 6:
            self.mw_classes = mw_glb2acls6
        elif num_classes == 3:
            self.mw_classes = mw_glb2bcls3
        else:
            raise ValueError
        # define model file
        self.fp_model = fp_model
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes
        self.ie = ie
        self.device = device

        # define conditions
        self.confthre = conf_thresh
        self.nmsthre = nms_thresh

    def get_openvino_model(self):
        net = self.ie.read_network(self.fp_model)

        input_name = next(iter(net.input_info))
        output_name = next(iter(net.outputs))
        net.input_info[input_name].precision = 'FP32'
        _, _, h, w = net.input_info[input_name].input_data.shape
        net.outputs[output_name].precision = 'FP32'

        return (
            input_name,
            output_name,
            (h, w),
            self.ie.load_network(network=net, device_name=self.device))
