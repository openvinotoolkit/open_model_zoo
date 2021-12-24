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
from openvino.inference_engine import IECore

sys.path.append('temporal_segmentation')
from multiview_mobilenetv3_tsm import create_mbv3s_model
# import mstcn


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

        output = self.classifier.infer(inputs={
            self.classifier_input_keys[0]: feature_vector_front,
            self.classifier_input_keys[1]: feature_vector_top}
            )[self.classifier_output_key[0]]

        ### yoclo classifier ###
        isAction = (output.squeeze()[0] >= .5).astype(int)
        predicted = isAction*(np.argmax(output.squeeze()[1:]) + 1)

        return self.terms[predicted], self.terms[predicted]


class SegmentorMstcn(Segmentor):
    def __init__(self, i3d_path, mstcn_path):
        self.embed_model = 0
        self.seg_model = 0
        self.temporal_predictions = 0

        self.i3d_path = i3d_path
        self.mstcn_path = mstcn_path

        self.ActionTerms = [
            "noise_action",
            "put_take",
            "adjust_rider"
            ]

        self.EmbedBufferTop = []
        self.EmbedBufferFront = []
        self.ImgSizeHeight  = 224
        self.ImgSizeWidth  = 224
        self.BatchSize = 1
        self.EnbedWindowLength = 16
        self.EnbedWindowStride = 1
        self.EnbedWindowAtrous = 3
        self.TemporalLogits = np.zeros((0, len(self.terms)))

    def initialize(self):
        ie = IECore()
        net = ie.read_network(
            model=self.i3d_path[:-4]+".xml",
            weights=self.i3d_path[:-4]+".bin"
        )
        self.i3d = ie.load_network(network=net, device_name="CPU")
        self.bi3d_input_keys =  list(self.i3d.input_info.keys())
        self.i3d_output_key = list(self.i3d.outputs.keys())
        net = ie.read_network(
            model=self.mstcn_path[:-4]+".xml",
            weights=self.mstcn_path[:-4]+".bin"
        )
        self.mstcn = ie.load_network(network=net, device_name="CPU")
        self.mstcn_input_keys =  list(self.mstcn .input_info.keys())
        self.mstcn_output_key = list(self.mstcn .outputs.keys())

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

        ### run encoder ###
        print("Frame embedding:", frame_index)
        self.EmbedBufferTop = self.feature_embedding(
            img_buffer=buffer_top,
            embedding_buffer=self.EmbedBufferTop,
            frame_index=frame_index)
        self.EmbedBufferFront = self.feature_embedding(
            img_buffer=buffer_front,
            embedding_buffer=self.EmbedBufferFront,
            frame_index=frame_index)

        ### run decoder wit vec 1x64x2048 ###
        # i3d_vec = np.zeros((1,64,2048)) 
        # output = self.mstcn.infer(inputs={
        #     self.mstcn_input_keys[0]: i3d_vec}
        #     )[self.mstcn_output_key[0]]

        # ### run tsmcn only batch size 1###
        # if min(self.EmbedBufferTop.shape[-1], self.EmbedBufferFront.shape[-1]) > 0:
        #     self.action_segmentation()

        # ### get label ###
        # isValid = self.TemporalLogits.shape[0]
        # if isValid:
        #     return [], []
        # else:
        #     frame_predictions = [self.ActionTerms[i] for i in np.argmax(self.TemporalLogits, axis=1)]
        #     frame_predictions = ["background" for i in range(self.EnbedWindowSize- 1)] + frame_predictions
        
        # return frame_predictions, frame_predictions

        return [], []

    def feature_embedding(self, img_buffer, embedding_buffer, frame_index):
        # minimal temporal length for processor
        min_t = 0 + self.EnbedWindowStride * 0 + (self.EnbedWindowLength - 1) * self.EnbedWindowAtrous

        if frame_index > min_t:
            num_embedding = embedding_buffer.shape[-1]
            img_buffer = list(img_buffer)
            feed_dict = {}

            curr_t = self.EnbedWindowStride * num_embedding + (self.EnbedWindowLength - 1) * self.EnbedWindowAtrous
            while curr_t < frame_index:
                # absolute index in temporal shaft
                start_index = self.EnbedWindowStride * num_embedding
                
                if frame_index > len(img_buffer):
                    # absolute index in buffer shaft
                    start_index = start_index - (frame_index - len(img_buffer))

                input_data = [
                    [cv2.resize(img_buffer[start_index + i * self.EnbedWindowAtrous],
                    (self.ImgSizeHeight, self.ImgSizeWidth)) for i in range(self.EnbedWindowLength)]
                    for j in range(self.BatchSize)]

                ###               ###
                ### inference i3d ###
                ###               ###
                out_logits, _ = self.i3d.infer(
                    inputs={self.backbone_input_keys[0]: input_data})[self.backbone_output_key[0]]
                embedding_buffer = np.concatenate([embedding_buffer, out_logits.T], axis=1)  # ndarray: C x num_embedding

                num_embedding += 1
                curr_t = self.EnbedWindowStride * num_embedding + (self.EnbedWindowLength - 1) * self.EnbedWindowAtrous


    def action_segmentation(self):
        # read buffer
        embed_buffer_top = self.EmbedBufferTop
        embed_buffer_front = self.EmbedBufferFront
        batch_size = self.BatchSize

        def softmax(x):
            """ applies softmax to an input x"""
            e_x = np.exp(x)
            return e_x / e_x.sum()

        start_index = self.temporal_logits.shape[0]
        end_index = min(embed_buffer_top.shape[-1], embed_buffer_front.shape[-1])
        num_batch = (end_index - start_index) // batch_size
        if num_batch < 0:
            print("Waiting for the next frame ...")
        elif num_batch == 0:
            print("start_index:",start_index,"end_index:",end_index)

            unit1 = embed_buffer_top[:, start_index:end_index]
            unit2 = embed_buffer_front[:, start_index:end_index]
            feature_unit = np.concatenate([unit1[:, ], unit2[:, ]], axis=0)
            input = feature_unit.unsqueeze_(0) # 2048xN

            predictions, self.his_fea = self.seg_model(input)
            """
                predictions --> 4x1x64x24
                his_fea --> [12*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048]]
            """
            temporal_logits = predictions[:, :, :len(self.terms), :]  # 4x1x16xN
            temporal_logits = softmax(temporal_logits[-1], 1)  # 1x16xN
            temporal_logits = temporal_logits.permute(0, 2, 1).squeeze(dim=0)
            self.TemporalLogits = np.concatenate([self.TemporalLogits, temporal_logits], axis=0)

        else:
            for batch_idx in range(num_batch):
                unit1 = embed_buffer_top[:,
                    start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                unit2 = embed_buffer_front[:,
                    start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                feature_unit = np.concatenate([unit1[:, ], unit2[:, ]], axis=0)
                input = feature_unit.unsqueeze_(0) # 2048x24

                # predictions, self.his_fea = self.seg_model(input, self.his_fea)
                predictions, self.his_fea = self.seg_model(input)

                temporal_logits = predictions[:, :, :len(self.terms), :]  # 4x1x16xN
                temporal_logits = softmax(temporal_logits[-1], 1)  # 1x16xN
                temporal_logits = temporal_logits.permute(0, 2, 1).squeeze(dim=0)
                self.TemporalLogits = np.concatenate([self.TemporalLogits, temporal_logits], axis=0)