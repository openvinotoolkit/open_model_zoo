# -*- coding: utf-8 -*-
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

import cv2
import numpy as np
import sys
sys.path.append('temporal_segmentation')
from multiview_mobilenetv3_tsm import create_mbv3s_model
from openvino.inference_engine import IECore

class Segmentor(object):
    def __init__(self, backbone_path, classifier_path):
        self.embed_model = 0
        self.seg_model = 0
        self.temporal_predictions = 0

        self.backbone_path = backbone_path
        self.classifier_path = classifier_path

        self.terms = [
            "noise_action",
            "put_take",
            "adjust_rider"
            ]

    def initialize(self):
        ie = IECore()
        net = ie.read_network(
            model=self.backbone_path[:-4]+".xml",
            weights=self.backbone_path[:-4]+".bin"
        )
        self.backbone = ie.load_network(network=net, device_name="CPU")
        self.backbone_input_keys =  list(self.backbone.input_info.keys())
        self.backbone_output_key = list(self.backbone.outputs.keys())

        net = ie.read_network(
            model=self.classifier_path[:-4]+".xml",
            weights=self.classifier_path[:-4]+".bin"
        )
        self.classifier = ie.load_network(network=net, device_name="CPU")
        self.classifier_input_keys =  list(self.classifier .input_info.keys())
        self.classifier_output_key = list(self.classifier .outputs.keys())

    def inference(self, buffer_top, buffer_front, frame_index):
        """
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_front: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()
        """

        ### preprocess ###
        buffer_front = cv2.resize(buffer_front,(224,224), interpolation= cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top,(224,224), interpolation= cv2.INTER_LINEAR)
        buffer_front = buffer_front/255
        buffer_top = buffer_top/255

        buffer_front = buffer_front[np.newaxis,:,:,:].transpose((0,3,1,2)).astype(np.float32)
        buffer_top = buffer_top[np.newaxis,:,:,:].transpose((0,3,1,2)).astype(np.float32)

        ### run ###
        feature_vector_front = self.backbone.infer(
            inputs={self.backbone_input_keys[0]: buffer_front})[self.backbone_output_key[0]]
        feature_vector_top = self.backbone.infer(
            inputs={self.backbone_input_keys[0]: buffer_top})[self.backbone_output_key[0]]
        output = [feature_vector_high, feature_vector_top]

        feature_vector_top = self.classifier.infer(
            inputs={self.classifier_input_keys[0]: features})[self.classifier_output_key[0]]

        ### yoclo classifier ###
        isAction = (output.squeeze()[0] >= .5).astype(int)
        predicted = (np.argmax(output.squeeze()[1:]) + 1)

        return self.terms[predicted], self.terms[predicted]
