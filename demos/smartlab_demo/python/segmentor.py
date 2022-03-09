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
from PIL import Image
from pathlib import Path
from scipy.special import softmax
from openvino.runtime import PartialShape


class Segmentor:
    def __init__(self, core, device, encoder_side_path, encoder_top_path, decoder_path):
        self.terms = [
            "noise_action",
            "put_take",
            "adjust_rider",
        ]

        net = core.read_model(encoder_side_path)
        self.encoder_side = core.compile_model(model=net, device_name=device)
        self.encoder_side_input_keys = self.encoder_side.inputs
        self.encoder_side_output_key = self.encoder_side.outputs
        net = core.read_model(encoder_top_path)
        self.encoder_top = core.compile_model(model=net, device_name=device)
        self.encoder_top_input_keys = self.encoder_top.inputs
        self.encoder_top_output_key = self.encoder_top.outputs
        net = core.read_model(decoder_path)
        self.decoder = core.compile_model(model=net, device_name=device)
        self.decoder_input_keys = self.decoder.inputs
        self.decoder_output_key = self.decoder.outputs

        self.shifted_tesor_side = np.zeros(85066)
        self.shifted_tesor_top = np.zeros(85066)

        ### infer request node ###
        self.infer_encoder_side_request = self.encoder_side.create_infer_request()
        self.infer_encoder_top_request = self.encoder_top.create_infer_request()
        self.infer_decoder_request = self.decoder.create_infer_request()

    def inference(self, buffer_top, buffer_side, frame_index):
        """
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_side: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()
        """

        ### preprocess ###
        buffer_side = buffer_side[120:, :, :] # remove date characters
        buffer_top = buffer_top[120:, :, :] # remove date characters
        buffer_side = cv2.resize(buffer_side, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_side = buffer_side / 255
        buffer_top = buffer_top / 255

        buffer_side = buffer_side[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)
        buffer_top = buffer_top[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)

        ### run ###
        feature_vector_side = self.infer_encoder_side_request.infer(
            inputs={self.encoder_side_input_keys[0]: buffer_side})[self.encoder_side_output_key[0]]
        feature_vector_top = self.infer_encoder_top_request.infer(
            inputs={self.encoder_top_input_keys[0]: buffer_top})[self.encoder_top_output_key[0]]
        output = self.infer_decoder_request.infer(inputs={
            self.decoder_input_keys[0]: feature_vector_side,
            self.decoder_input_keys[1]: feature_vector_top}
        )[self.decoder_output_key[0]]

        ### yoclo classifier ###
        isAction = (output.squeeze()[0] >= .5).astype(int)
        predicted = isAction * (np.argmax(output.squeeze()[1:]) + 1)

        return self.terms[predicted], self.terms[predicted]

    def inference_async(self, buffer_top, buffer_side, frame_index):
        ### preprocess ###
        buffer_side = buffer_side[120:, :, :]  # remove date characters
        buffer_top = buffer_top[120:, :, :]  # remove date characters
        buffer_side = cv2.resize(buffer_side, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_side = buffer_side / 255
        buffer_top = buffer_top / 255

        buffer_side = buffer_side[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)
        buffer_top = buffer_top[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)

        self.infer_encoder_side_request.start_async(inputs=
            {self.encoder_side_input_keys[0]: buffer_side,
            self.encoder_side_input_keys[1]: self.shifted_tesor_side})

        self.infer_encoder_top_request.start_async(inputs=
            {self.encoder_top_input_keys[0]: buffer_top,
            self.encoder_top_input_keys[1]: self.shifted_tesor_top})

        while True:
            if self.infer_encoder_side_request.wait_for(0) and self.infer_encoder_top_request.wait_for(0):
                feature_vector_side = self.infer_encoder_side_request.get_tensor(
                    self.encoder_side_output_key[0])
                self.shifted_tesor_side = self.infer_encoder_side_request.get_tensor(
                    self.encoder_side_output_key[1]).data
                feature_vector_top = self.infer_encoder_top_request.get_tensor(
                    self.encoder_top_output_key[0])
                self.shifted_tesor_top = self.infer_encoder_top_request.get_tensor(
                    self.encoder_top_output_key[1]).data

                output = self.infer_decoder_request.infer(inputs={
                    self.decoder_input_keys[0]: feature_vector_side.data,
                    self.decoder_input_keys[1]: feature_vector_top.data})[self.decoder_output_key[0]]

                ### yoclo classifier ###
                isAction = (output.squeeze()[0] >= .5).astype(int)
                predicted = isAction * (np.argmax(output.squeeze()[1:]) + 1)

                return self.terms[predicted], self.terms[predicted]

class SegmentorMstcn:
    def __init__(self, core, device, encoder_path, mstcn_path):
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

        self.EmbedBufferTop = np.zeros((1280, 0))
        self.EmbedBufferSide = np.zeros((1280, 0))
        self.ImgSizeHeight = 224
        self.ImgSizeWidth = 224

        self.SegBatchSize = 24
        self.EmbedWindowLength = 1
        self.TemporalLogits = np.zeros((0, len(self.ActionTerms)))

        # efficientnetB0
        net = core.read_model(encoder_path)
        nodes = net.get_ops()
        net.add_outputs(nodes[13].output(0))
        self.efficientNet = core.compile_model(model=net, device_name=device)
        self.efficientNet_input_keys = self.efficientNet.inputs
        self.efficientNet_output_key = self.efficientNet.outputs
        self.efficientNet_top_request = self.efficientNet.create_infer_request()
        self.efficientNet_side_request = self.efficientNet.create_infer_request()

        self.mstcn_net = core.read_model(mstcn_path)
        self.mstcn = core.compile_model(model=self.mstcn_net, device_name=device)
        self.mstcn_input_keys = self.mstcn.inputs
        self.mstcn_output_keys = self.mstcn.outputs
        self.mstcn_net.reshape({'input': PartialShape([1, 2560, 1])})
        self.reshape_mstcn = core.compile_model(model=self.mstcn_net, device_name=device)
        self.mstcn_infer_request = self.reshape_mstcn.create_infer_request()

        file_path = Path(__file__).parent / 'init_his.npz'
        init_his_feature = np.load(file_path)
        self.his_fea = {f'fhis_in_{i}': init_his_feature[f'arr_{i}'] for i in range(4)}

    def inference(self, frame_top, frame_side, frame_index):
        """
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_side: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()
        """
        ### run encoder ###
        self.feature_embedding(
            img=frame_top, view="top")
        self.feature_embedding(
            img=frame_side, view="side")

        while True:
            if self.efficientNet_top_request.wait_for(0) and self.efficientNet_side_request.wait_for(0):
                top_feature = self.efficientNet_top_request.get_tensor(
                    self.efficientNet_output_key[1]).data.squeeze(axis=2).squeeze(axis=2).transpose(1, 0)
                side_feature = self.efficientNet_side_request.get_tensor(
                    self.efficientNet_output_key[1]).data.squeeze(axis=2).squeeze(axis=2).transpose(1, 0)

                self.EmbedBufferTop = np.concatenate((self.EmbedBufferTop, top_feature), axis=1)
                self.EmbedBufferSide = np.concatenate((self.EmbedBufferSide, side_feature), axis=1)

                ### run mstcn++ ###
                self.action_segmentation(frame_index)

                # ### get label ###
                valid_index = self.TemporalLogits.shape[0]
                if valid_index == 0:
                    return []
                else:
                    frame_predictions = [self.ActionTerms[i] for i in np.argmax(self.TemporalLogits, axis=1)]
                    frame_predictions = ["background" for i in range(self.EmbedWindowLength - 1)] + frame_predictions

                return frame_predictions[-1]

    def feature_embedding(self, img, view):
        input_data = np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
        input_data = np.asarray(input_data).squeeze().transpose(2, 0, 1)

        if view == "top":
            self.efficientNet_top_request.start_async(
                inputs={self.efficientNet_input_keys[0]: [input_data]})
        else:
            self.efficientNet_side_request.start_async(
                inputs={self.efficientNet_input_keys[0]: [input_data]})

    def action_segmentation(self, frame_index):
        # read buffer
        embed_buffer_top = self.EmbedBufferTop # (1280x1)
        embed_buffer_side = self.EmbedBufferSide # (1280x1)
        batch_size = self.SegBatchSize
        start_index = self.TemporalLogits.shape[0]
        end_index = min(embed_buffer_top.shape[-1], embed_buffer_side.shape[-1])
        num_batch = (end_index - start_index) // batch_size

        if num_batch < 0:
            log.debug("Waiting for the next frame ...")
        elif num_batch == 0: 
            log.debug(f"start_index: {start_index} end_index: {end_index}")
            unit1 = embed_buffer_top
            unit2 = embed_buffer_side
            feature_unit = np.concatenate((unit1, unit2), axis=0)
            input_mstcn = np.expand_dims(feature_unit, 0)

            feed_dict = {}
            if len(self.his_fea) != 0:
                for key in self.mstcn_input_keys:
                    if 'fhis_in_' in str(key.names):
                        string = list(key.names)[0]
                        feed_dict[string] = self.his_fea[string]
            feed_dict['input'] = input_mstcn

            # inference MSTCN
            if input_mstcn.shape == (1, 2560, 1):
                self.mstcn_infer_request.infer(feed_dict)

            # get predicted output
            predictions = self.mstcn_infer_request.get_tensor(self.mstcn_output_keys[0]).data

            for key in self.mstcn_input_keys[1:]:
                if 'fhis_in_' in str(key.names): # fhis_in_0 to 3
                    string = list(key.names)[0]
                    self.his_fea[string] = self.mstcn_infer_request.get_tensor(key).data

            pred_actions = predictions[:, :, :len(self.ActionTerms), :]  # 4x1x16xN
            pred_softmax = softmax(pred_actions[-1], 1)  # 1x16xN
            temporal_logits = pred_softmax.transpose((0, 2, 1)).squeeze(axis=0)
            self.TemporalLogits = np.concatenate((self.TemporalLogits, temporal_logits), axis=0)
        else:
            raise ValueError(f"unexpected batch size: {num_batch}")