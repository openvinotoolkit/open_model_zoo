"""
 Copyright (c) 2021-2024 Intel Corporation

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
import copy
import numpy as np

from scipy.special import softmax


class Segmentor:
    def __init__(self, core, device, encoder_side_path, encoder_top_path, decoder_path):
        self.terms = [
            "noise_action",
            "put_take",
            "adjust_rider",
        ]

        # side encoder
        net = core.read_model(encoder_side_path)
        self.encoder_side = core.compile_model(model=net, device_name=device)
        self.encoder_side_input_keys = {list(item.get_names())[0]: item for item in self.encoder_side.inputs}
        self.encoder_side_output_keys = {list(item.get_names())[0]: item for item in self.encoder_side.outputs}

        # top encoder
        net = core.read_model(encoder_top_path)
        self.encoder_top = core.compile_model(model=net, device_name=device)
        self.encoder_top_input_keys = {list(item.get_names())[0]: item for item in self.encoder_top.inputs}
        self.encoder_top_output_keys = {list(item.get_names())[0]: item for item in self.encoder_top.outputs}

        # dencoder
        net = core.read_model(decoder_path)
        self.decoder = core.compile_model(model=net, device_name=device)
        self.decoder_input_keys = {list(item.get_names())[0]: item for item in self.decoder.inputs}
        self.decoder_output_key = {list(item.get_names())[0]: item for item in self.decoder.outputs}

        self.shifted_tesor_side = np.zeros(85066)
        self.shifted_tesor_top = np.zeros(85066)

        ### infer request node ###
        self.infer_encoder_side_request = self.encoder_side.create_infer_request()
        self.infer_encoder_top_request = self.encoder_top.create_infer_request()
        self.infer_decoder_request = self.decoder.create_infer_request()

    def inference(self, frame_top, frame_side, frame_index):
        """
        Args:
            frame_top: buffers of the input image arrays for the top view
            frame_side: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()
        """

        ### preprocess ###
        frame_side = frame_side[120:, :, :]  # remove date characters
        frame_top = frame_top[120:, :, :]  # remove date characters
        frame_side = cv2.resize(frame_side, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame_top = cv2.resize(frame_top, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame_side = frame_side / 255
        frame_top = frame_top / 255

        frame_side = frame_side[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)
        frame_top = frame_top[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)

        ### run ###
        self.infer_encoder_side_request.infer(inputs=
                                              {self.encoder_side_input_keys['input_image']: frame_side,
                                               self.encoder_side_input_keys['shifted_input']: self.shifted_tesor_side})

        self.infer_encoder_top_request.infer(inputs=
                                             {self.encoder_top_input_keys['input_image']: frame_top,
                                              self.encoder_top_input_keys['shifted_input']: self.shifted_tesor_top})

        ### get tensors ###
        feature_vector_side = self.infer_encoder_side_request.get_tensor(
            self.encoder_side_output_keys['output_feature'])
        self.shifted_tesor_side = self.infer_encoder_side_request.get_tensor(
            self.encoder_side_output_keys['shifted_output']).data

        feature_vector_top = self.infer_encoder_top_request.get_tensor(
            self.encoder_top_output_keys['output_feature'])
        self.shifted_tesor_top = self.infer_encoder_top_request.get_tensor(
            self.encoder_top_output_keys['shifted_output']).data

        output = self.infer_decoder_request.infer(inputs={
            self.decoder_input_keys['input_feature_1']: feature_vector_side.data,
            self.decoder_input_keys['input_feature_2']: feature_vector_top.data})[self.decoder_output_key['output']]

        ### yoclo classifier ###
        isAction = (output.squeeze()[0] >= .5).astype(int)
        predicted = isAction * (np.argmax(output.squeeze()[1:]) + 1)

        return self.terms[predicted], self.terms[predicted], frame_index

    def inference_async(self, frame_top, frame_side, frame_index):
        ### preprocess ###
        frame_side = frame_side[120:, :, :]  # remove date characters
        frame_top = frame_top[120:, :, :]  # remove date characters
        frame_side = cv2.resize(frame_side, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame_top = cv2.resize(frame_top, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame_side = frame_side / 255
        frame_top = frame_top / 255

        frame_side = frame_side[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)
        frame_top = frame_top[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)

        self.infer_encoder_side_request.start_async(inputs={
            self.encoder_side_input_keys['input_image']: frame_side,
            self.encoder_side_input_keys['shifted_input']: self.shifted_tesor_side})

        self.infer_encoder_top_request.start_async(inputs={self.encoder_top_input_keys['input_image']: frame_top,
                                                           self.encoder_top_input_keys['shifted_input']: self.shifted_tesor_top})

        while True:
            if self.infer_encoder_side_request.wait_for(0) and self.infer_encoder_top_request.wait_for(0):
                feature_vector_side = self.infer_encoder_side_request.get_tensor(
                    self.encoder_side_output_keys['output_feature'])
                self.shifted_tesor_side = self.infer_encoder_side_request.get_tensor(
                    self.encoder_side_output_keys['shifted_output']).data
                feature_vector_top = self.infer_encoder_top_request.get_tensor(
                    self.encoder_top_output_keys['output_feature'])
                self.shifted_tesor_top = self.infer_encoder_top_request.get_tensor(
                    self.encoder_top_output_keys['shifted_output']).data

                output = self.infer_decoder_request.infer(inputs={
                    self.decoder_input_keys['input_feature_1']: feature_vector_side.data,
                    self.decoder_input_keys['input_feature_2']: feature_vector_top.data})[
                    self.decoder_output_key['output']]

                ### yoclo classifier ###
                isAction = (output.squeeze()[0] >= .5).astype(int)
                predicted = isAction * (np.argmax(output.squeeze()[1:]) + 1)

                return self.terms[predicted], self.terms[predicted], frame_index


class SegmentorMstcn:
    def __init__(self, core, device, encoder_path, mstcn_path):
        self.ActionTerms = [
            "noise_action",
            "background",
            "remove_support_sleeve",
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
        ]

        self.ImgSizeHeight = 224
        self.ImgSizeWidth = 224
        self.SegBatchSize = 24
        self.EmbedBufferCombined = []

        # mobilenet-v3-small
        net = core.read_model(encoder_path)
        # set batch size 2 to make it accept top and front view at the same time
        net.reshape([2, 3, 224, 224])
        self.mobileNet = core.compile_model(model=net, device_name=device)
        self.mobileNet_input_keys = self.mobileNet.inputs
        self.mobileNet_output_key = self.mobileNet.outputs
        self.mobileNet_request = self.mobileNet.create_infer_request()

        self.mstcn_net = core.read_model(mstcn_path)
        self.mstcn_net.reshape({"input": [1, 1152, self.SegBatchSize]})
        self.mstcn = core.compile_model(model=self.mstcn_net, device_name=device)
        self.mstcn_input_keys = self.mstcn.inputs
        self.mstcn_output_keys = self.mstcn.outputs
        self.reshape_mstcn = core.compile_model(model=self.mstcn_net, device_name=device)
        self.mstcn_infer_request = self.reshape_mstcn.create_infer_request()
        self.his_fea = [np.zeros((12, 64, 2048)), np.zeros((11, 64, 2048)),
                        np.zeros((11, 64, 2048)), np.zeros((11, 64, 2048))]

    def inference(self, frame_top, frame_side, frame_index):
        """
        Args:
            frame_top: buffers of the input image arrays for the top view
            frame_side: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()
        """
        ### run mobilenet ###
        self.feature_embedding(frame_top, frame_side)
        feature = self.mobileNet_request.get_tensor(self.mobileNet_output_key[0]).data.reshape(1152, 1)
        # ### run mstcn++ ###
        return self.action_segmentation(feature)

    def feature_embedding(self, frame_top, frame_side):
        img_top = cv2.resize(frame_top, (224, 224)) / 255.0
        img_side = cv2.resize(frame_side, (224, 224)) / 255.0
        combined_img = np.concatenate(
            (np.expand_dims(img_top, axis=0), np.expand_dims(img_side, axis=0)),
            axis=0).transpose((0, 3, 1, 2))
        self.mobileNet_request.infer(inputs={self.mobileNet_input_keys[0]: combined_img})

    def action_segmentation(self, feature):
        # add up feature
        feature = copy.copy(feature)
        self.EmbedBufferCombined.append(feature)
        if len(self.EmbedBufferCombined) == self.SegBatchSize:
            input_mstcn = np.asarray(self.EmbedBufferCombined).transpose(2, 1, 0)
            # reset bufferCombined
            self.EmbedBufferCombined.clear()
            feed_dict = {'input': input_mstcn, 'fhis_in_0': self.his_fea[0], 'fhis_in_1': self.his_fea[1],
                         'fhis_in_2': self.his_fea[2], 'fhis_in_3': self.his_fea[3]}

            # inference MSTCN
            self.mstcn_infer_request.infer(feed_dict)

            # get predicted output
            predictions = self.mstcn_infer_request.get_tensor(self.mstcn_output_keys[0]).data

            for i in range(4):
                self.his_fea[i] = self.mstcn_infer_request.get_tensor(self.mstcn_output_keys[i + 1]).data

            pred_actions = predictions[:, :, :len(self.ActionTerms), :]  # 4x3xKxN
            pred_softmax = softmax(pred_actions[-1, 0], 0)  # KxN
            temporal_logits = pred_softmax.transpose((1, 0))  # NxK

            ### get label ###
            # 0 - noise_action: others
            # 1 - put_take: [9,10,11,12]
            # 2 - adjust_rider: [3]
            frame_predictions = []
            for i in np.argmax(temporal_logits, axis=1):
                if i == 3:
                    frame_predictions.append("adjust_rider")
                elif i == 9 or i == 10 or i == 11 or i == 12:
                    frame_predictions.append("put_take")
                else:
                    frame_predictions.append("noise_action")
            frame_predictions.append(self.ActionTerms[i])
            # frame_predictions = [self.ActionTerms[i] for i in action_idx] # N

            return frame_predictions
