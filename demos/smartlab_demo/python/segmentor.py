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
import logging as log
from collections import deque
from scipy.special import softmax
from openvino.inference_engine import IECore


class Segmentor(object):
    def __init__(self, ie, device, backbone_path, classifier_path):
        self.backbone_path = backbone_path
        self.classifier_path = classifier_path

        self.terms = [
            "noise_action",
            "put_take",
            "adjust_rider",
        ]

        net = ie.read_network(self.backbone_path)
        self.backbone = ie.load_network(network=net, device_name=device)
        self.backbone_input_keys = list(self.backbone.input_info.keys())
        self.backbone_output_key = list(self.backbone.outputs.keys())
        net = ie.read_network(self.classifier_path)
        self.classifier = ie.load_network(network=net, device_name=device)
        self.classifier_input_keys = list(self.classifier.input_info.keys())
        self.classifier_output_key = list(self.classifier.outputs.keys())


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
        buffer_front = buffer_front[120:, :, :] # remove date characters
        buffer_top = buffer_top[120:, :, :] # remove date characters
        buffer_front = cv2.resize(buffer_front, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_front = buffer_front / 255
        buffer_top = buffer_top / 255

        buffer_front = buffer_front[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)
        buffer_top = buffer_top[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)

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
        predicted = isAction * (np.argmax(output.squeeze()[1:]) + 1)

        return self.terms[predicted], self.terms[predicted]


class SegmentorMstcn(object):
    def __init__(self, ie, device, i3d_path, mstcn_path):
        self.embed_model = 0
        self.seg_model = 0
        self.temporal_predictions = 0

        self.i3d_path = i3d_path
        self.mstcn_path = mstcn_path

        self.ActionTerms = [
            "background",
            "noise_action",
            "remove_support_sleeve",
            "remove_pointer_sleeve",
            "adjust_rider",
            "adjust_nut",
            "adjust_balancing",
            "open_box",
            "close_box",
            "choose_weight",
            "put_left",
            "put_right",
            "take_left",
            "take_right",
            "install_support_sleeve",
            "install_pointer_sleeve",
        ]

        self.EmbedBufferTop = np.zeros((1024, 0))
        self.EmbedBufferFront = np.zeros((1024, 0))
        self.ImgSizeHeight = 224
        self.ImgSizeWidth = 224
        self.EmbedBatchSize = 1
        self.SegBatchSize = 24
        self.EmbedWindowLength = 16
        self.EmbedWindowStride = 1
        self.EmbedWindowAtrous = 3
        self.TemporalLogits = np.zeros((0, len(self.ActionTerms)))
        self.his_fea = []

        net = ie.read_network(self.i3d_path)
        net.reshape({next(iter(net.input_info)): (
            self.EmbedBatchSize, 3, self.EmbedWindowLength, self.ImgSizeWidth, self.ImgSizeHeight)})

        net.add_outputs("RGB/inception_i3d/Logits/AvgPool3D")

        self.i3d = ie.load_network(network=net, device_name=device)
        self.i3d_input_keys = list(self.i3d.input_info.keys())
        self.i3d_output_key = list(self.i3d.outputs.keys())

        self.mstcn_net = ie.read_network(self.mstcn_path)
        self.mstcn = ie.load_network(network=self.mstcn_net, device_name=device)
        self.mstcn_input_keys = list(self.mstcn.input_info.keys())
        self.mstcn_output_key = list(self.mstcn.outputs.keys())
        self.mstcn_net.reshape({'input': (1, 2048, 1)})
        self.reshape_mstcn = ie.load_network(network=self.mstcn_net, device_name=device)
        init_his_feature = np.load('init_his.npz')
        self.his_fea = [init_his_feature['arr_0'],
                init_his_feature['arr_1'],
                init_his_feature['arr_2'],
                init_his_feature['arr_3']]


    def inference(self, buffer_top, buffer_front, frame_index):
        """
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_front: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()
        """
        ### run encoder ###
        self.EmbedBufferTop = self.feature_embedding(
            img_buffer=buffer_top,
            embedding_buffer=self.EmbedBufferTop,
            frame_index=frame_index)
        self.EmbedBufferFront = self.feature_embedding(
            img_buffer=buffer_front,
            embedding_buffer=self.EmbedBufferFront,
            frame_index=frame_index)

        ### run mstcn++ only batch size 1###
        if min(self.EmbedBufferTop.shape[-1], self.EmbedBufferFront.shape[-1]) > 0:
            self.action_segmentation()

        # ### get label ###
        valid_index = self.TemporalLogits.shape[0]
        if valid_index == 0:
            return []
        else:
            frame_predictions = [self.ActionTerms[i] for i in np.argmax(self.TemporalLogits, axis=1)]
            frame_predictions = ["background" for i in range(self.EmbedWindowLength - 1)] + frame_predictions

        return frame_predictions[-1]

    def feature_embedding(self, img_buffer, embedding_buffer, frame_index):
        # minimal temporal length for processor
        min_t = (self.EmbedWindowLength - 1) * self.EmbedWindowAtrous

        if frame_index > min_t:
            num_embedding = embedding_buffer.shape[-1]
            img_buffer = list(img_buffer)
            curr_t = self.EmbedWindowStride * num_embedding + (self.EmbedWindowLength - 1) * self.EmbedWindowAtrous
            while curr_t < frame_index:
                # absolute index in temporal shaft
                start_index = self.EmbedWindowStride * num_embedding

                if frame_index > len(img_buffer):
                    # absolute index in buffer shaft
                    start_index = start_index - (frame_index - len(img_buffer))

                input_data = [
                    [cv2.resize(img_buffer[start_index + i * self.EmbedWindowAtrous],
                                (self.ImgSizeHeight, self.ImgSizeWidth)) for i in range(self.EmbedWindowLength)]
                    for j in range(self.EmbedBatchSize)]
                input_data = np.asarray(input_data).transpose((0, 4, 1, 2, 3))
                input_data = input_data * 127.5 + 127.5

                out_logits = self.i3d.infer(
                    inputs={self.i3d_input_keys[0]: input_data})[self.i3d_output_key[0]]
                out_logits = out_logits.squeeze((0, 3, 4))
                embedding_buffer = np.concatenate([embedding_buffer, out_logits],
                                                  axis=1)  # ndarray: C x num_embedding

                curr_t += self.EmbedWindowStride
        return embedding_buffer

    def action_segmentation(self):
        # read buffer
        embed_buffer_top = self.EmbedBufferTop
        embed_buffer_front = self.EmbedBufferFront
        batch_size = self.SegBatchSize
        start_index = self.TemporalLogits.shape[0]
        end_index = min(embed_buffer_top.shape[-1], embed_buffer_front.shape[-1])
        num_batch = (end_index - start_index) // batch_size
        if num_batch < 0:
            log.debug("Waiting for the next frame ...")
        elif num_batch == 0:
            log.debug(f"start_index: {start_index} end_index: {end_index}")

            unit1 = embed_buffer_top[:, start_index:end_index]
            unit2 = embed_buffer_front[:, start_index:end_index]
            feature_unit = np.concatenate([unit1[:, ], unit2[:, ]], axis=0)
            input_mstcn = np.expand_dims(feature_unit, 0)

            feed_dict = {}
            if len(self.his_fea) != 0:
                feed_dict = {self.mstcn_input_keys[i]: self.his_fea[i] for i in range(4)}
            feed_dict[self.mstcn_input_keys[-1]] = input_mstcn
            if input_mstcn.shape == (1, 2048, 1):
                out = self.reshape_mstcn.infer(inputs=feed_dict)

            predictions = out[self.mstcn_output_key[-1]]
            self.his_fea = [out[self.mstcn_output_key[i]] for i in range(4)]

            """
                predictions --> 4x1x64x24
                his_fea --> [12*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048]]
            """
            temporal_logits = predictions[:, :, :len(self.ActionTerms), :]  # 4x1x16xN
            temporal_logits = softmax(temporal_logits[-1], 1)  # 1x16xN
            temporal_logits = temporal_logits.transpose((0, 2, 1)).squeeze(axis=0)
            self.TemporalLogits = np.concatenate([self.TemporalLogits, temporal_logits], axis=0)
        else:
            for batch_idx in range(num_batch):
                unit1 = embed_buffer_top[:,
                        start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                unit2 = embed_buffer_front[:,
                        start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                feature_unit = np.concatenate([unit1[:, ], unit2[:, ]], axis=0)
                feed_dict = {}
                if len(self.his_fea) != 0:
                    feed_dict = {self.mstcn_input_keys[i]: self.his_fea[i] for i in range(4)}
                feed_dict[self.mstcn_input_keys[-1]] = feature_unit
                out = self.mstcn.infer(inputs=feed_dict)
                predictions = out[self.mstcn_output_key[-1]]
                self.his_fea = [out[self.mstcn_output_key[i]] for i in range(4)]

                temporal_logits = predictions[:, :, :len(self.ActionTerms), :]  # 4x1x16xN
                temporal_logits = softmax(temporal_logits[-1], 1)  # 1x16xN
                temporal_logits = temporal_logits.transpose((0, 2, 1)).squeeze(axis=0)
                self.TemporalLogits = np.concatenate([self.TemporalLogits, temporal_logits], axis=0)
