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
from pathlib import Path
from scipy.special import softmax
from openvino.runtime import PartialShape

class SegmentorMstcn:
    def __init__(self, core, device, i3d_path, mstcn_path):
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

        net = core.read_model(i3d_path)
        net.reshape({net.inputs[0]: PartialShape(
            [self.EmbedBatchSize, self.EmbedWindowLength, self.ImgSizeHeight, self.ImgSizeWidth, 3])})
        nodes = net.get_ops()
        net.add_outputs(nodes[11].output(0))
        self.i3d = core.compile_model(model=net, device_name=device)

        self.mstcn_net = core.read_model(mstcn_path)
        self.mstcn = core.compile_model(model=self.mstcn_net, device_name=device)
        self.mstcn_input_keys = self.mstcn.inputs
        self.mstcn_output_key = self.mstcn.outputs
        self.mstcn_net.reshape({'input': PartialShape([1, 2048, 1])})
        self.reshape_mstcn = core.compile_model(model=self.mstcn_net, device_name=device)
        file_path = Path(__file__).parent / 'init_his.npz'
        init_his_feature = np.load(file_path)
        self.his_fea = {f'fhis_in_{i}': init_his_feature[f'arr_{i}'] for i in range(4)}

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

        infer_request = self.i3d.create_infer_request()
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
                input_data = np.asarray(input_data) * 127.5 + 127.5

                input_dict = {self.i3d.inputs[0]: input_data}
                out_logits = infer_request.infer(input_dict)[self.i3d.outputs[1]]
                out_logits = out_logits.squeeze((0, 3, 4))

                # ndarray: C x num_embedding
                embedding_buffer = np.concatenate((embedding_buffer, out_logits), axis=1)

                curr_t += self.EmbedWindowStride
        return embedding_buffer

    def action_segmentation(self):
        # read buffer
        batch_size = self.SegBatchSize
        embed_buffer_top = self.EmbedBufferTop
        embed_buffer_front = self.EmbedBufferFront
        start_index = self.TemporalLogits.shape[0]
        end_index = min(embed_buffer_top.shape[-1], embed_buffer_front.shape[-1])
        num_batch = (end_index - start_index) // batch_size

        infer_request = self.reshape_mstcn.create_infer_request()

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
                for key in self.mstcn_input_keys:
                    if 'fhis_in_' in str(key.names):
                        string = list(key.names)[0]
                        feed_dict[string] = self.his_fea[string]
            feed_dict['input'] = input_mstcn

            # if the input shape is unsupported shape, add dummpy result of noise and skip inference
            if input_mstcn.shape != (1, 2048, 1):
                tranposed_logits = np.zeros((2048, 16))
                self.TemporalLogits = np.concatenate([self.TemporalLogits, tranposed_logits], axis=0)
                return None

            out = infer_request.infer(feed_dict)
            predictions = out[list(out.keys())[-1]]
            for key in self.mstcn_output_key:
                if 'fhis_in_' in str(key.names):
                    string = list(key.names)[0]
                    self.his_fea[string] = out[string]

            """
                predictions --> 4x64x24
                his_fea --> [12*[64x2048], 11*[64x2048], 11*[64x2048], 11*[64x2048]]
            """
            temporal_logits = predictions[:, :len(self.ActionTerms), :] # Nx16x2048
            softmaxed_logits = softmax(temporal_logits[-1], 1) # 16x2048
            tranposed_logits = softmaxed_logits.transpose((1, 0)) # 2048x16
            self.TemporalLogits = np.concatenate([self.TemporalLogits, tranposed_logits], axis=0)
        else:
            for batch_idx in range(num_batch):
                unit1 = embed_buffer_top[:,
                        start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                unit2 = embed_buffer_front[:,
                        start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                feature_unit = np.concatenate([unit1[:, ], unit2[:, ]], axis=0)

                feed_dict = {}
                if len(self.his_fea) != 0:
                    for key in self.mstcn_input_keys:
                        if 'fhis_in_' in str(key.names):
                            string = list(key.names)[0]
                            feed_dict[key] = self.his_fea[string]
                feed_dict['input'] = feature_unit
                out = infer_request.infer(feed_dict)
                predictions = out[list(out.keys())[-1]]
                for key in self.mstcn_output_key:
                    if 'fhis_in_' in str(key.names):
                        string = list(key.names)[0]
                        self.his_fea[string] = out[string]

                temporal_logits = predictions[:, :len(self.ActionTerms), :]
                softmaxed_logits = softmax(temporal_logits[-1], 1)
                tranposed_logits = softmaxed_logits.transpose((1, 0))
                self.TemporalLogits = np.concatenate([self.TemporalLogits, temporal_logits], axis=0)
