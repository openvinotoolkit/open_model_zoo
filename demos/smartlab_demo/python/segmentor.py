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
