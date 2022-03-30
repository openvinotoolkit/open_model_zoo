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
    def __init__(self, num_classes, fp_model, nms_thresh, conf_thresh, core, device):
        if num_classes == 10:
            self.mw_classes = mw_glb1cls10
        elif num_classes == 3:
            self.mw_classes = mw_glb2bcls3
        else:
            raise ValueError(f'num_classes={num_classes} is not supported, use 10 or 3')
        # define model file
        self.fp_model = fp_model
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes
        self.core = core
        self.device = device

    def get_openvino_model(self):
        model = self.core.read_model(self.fp_model)
        compiled_model = self.core.compile_model(model, self.device)
        input_name = compiled_model.inputs[0]
        output_name = compiled_model.outputs[0]

        return (input_name, output_name, (416, 416), compiled_model)
