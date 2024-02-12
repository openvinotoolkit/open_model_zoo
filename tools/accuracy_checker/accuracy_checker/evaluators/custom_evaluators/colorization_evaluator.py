"""
Copyright (c) 2018-2024 Intel Corporation

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

import numpy as np
import cv2

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseDLSDKModel, BaseCascadeModel, BaseOpenVINOModel, create_model
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import extract_image_representations, contains_all, parse_partial_shape


class ColorizationEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)

        model = ColorizationCascadeModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )

        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            extr_batch_inputs, _ = extract_image_representations(batch_inputs)
            metrics_result = None
            batch_raw_prediction, batch_out = self.model.test_model.predict(batch_identifiers, extr_batch_inputs)
            if output_callback:
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            batch_raw_prediction, batch_prediction = self.model.check_model.predict(batch_identifiers, batch_out)
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


class ColorizationCascadeModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['colorization_network', 'verification_network']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('configuration for colorization_network/verification_network does not exist')

        self._test_mapping = {
            'dlsdk': ColorizationTestModel,
            'openvino': ColorizationTestOVModel
        }
        self._check_mapping = {
            'dlsdk': ColorizationCheckModel,
            'openvino': ColorizationCheckOVModel
        }
        self.test_model = create_model(network_info.get('colorization_network', {}), launcher, self._test_mapping,
                                       'colorization_network', delayed_model_loading)
        self.check_model = create_model(network_info.get('verification_network', {}), launcher, self._check_mapping,
                                        'verification_network', delayed_model_loading)
        self._part_by_name = {'colorization_network': self.test_model, 'verification_network': self.check_model}

    @property
    def adapter(self):
        return self.check_model.adapter

    def predict(self, identifiers, input_data, encoder_callback=None):
        pass


class ColorizationTestModel(BaseDLSDKModel):
    @staticmethod
    def data_preparation(input_data):
        input_ = input_data[0].astype(np.float32)
        img_lab = cv2.cvtColor(input_, cv2.COLOR_RGB2Lab)
        img_l = np.copy(img_lab[:, :, 0])
        img_l_rs = np.copy(img_lab[:, :, 0])
        return img_l, img_l_rs

    @staticmethod
    def central_crop(input_data, crop_size=(224, 224)):
        h, w = input_data.shape[:2]
        delta_h = (h - crop_size[0]) // 2
        delta_w = (w - crop_size[1]) // 2
        return input_data[delta_h:h - delta_h, delta_w: w - delta_w, :]

    def postprocessing(self, res, img_l):
        res = np.squeeze(res, axis=0)
        res = res.transpose((1, 2, 0)).astype(np.float32)

        out_lab = np.concatenate((img_l[:, :, np.newaxis], res), axis=2)
        result_bgr = np.clip(cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR), 0, 1)

        return [self.central_crop(result_bgr)]

    def predict(self, identifiers, input_data):
        img_l, img_l_rs = self.data_preparation(input_data)

        self.inputs[self.input_blob] = img_l_rs
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in self.inputs.items()})
        res = self.exec_network.infer(inputs=self.inputs)

        new_result = self.postprocessing(res[self.output_blob], img_l)
        return res, np.array(new_result)

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        super().set_input_and_output()
        self.inputs = {}
        for input_name in input_info:
            self.inputs[input_name] = np.zeros(input_info[input_name].input_data.shape)


class ColorizationTestOVModel(BaseOpenVINOModel):
    @staticmethod
    def data_preparation(input_data):
        input_ = input_data[0].astype(np.float32)
        img_lab = cv2.cvtColor(input_, cv2.COLOR_RGB2Lab)
        img_l = np.copy(img_lab[:, :, 0])
        img_l_rs = np.copy(img_lab[:, :, 0])
        return img_l, img_l_rs

    @staticmethod
    def central_crop(input_data, crop_size=(224, 224)):
        h, w = input_data.shape[:2]
        delta_h = (h - crop_size[0]) // 2
        delta_w = (w - crop_size[1]) // 2
        return input_data[delta_h:h - delta_h, delta_w: w - delta_w, :]

    def postprocessing(self, res, img_l):
        res = np.squeeze(res, axis=0)
        res = res.transpose((1, 2, 0)).astype(np.float32)

        out_lab = np.concatenate((img_l[:, :, np.newaxis], res), axis=2)
        result_bgr = np.clip(cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR), 0, 1)

        return [self.central_crop(result_bgr)]

    def predict(self, identifiers, input_data):
        img_l, img_l_rs = self.data_preparation(input_data)

        self._inputs[self.input_blob] = img_l_rs
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in self._inputs.items()})
        res, raw_res = self.infer(self._inputs, raw_results=True)

        new_result = self.postprocessing(res[self.output_blob], img_l)
        return raw_res, np.array(new_result)

    def set_input_and_output(self):
        super().set_input_and_output()
        self._inputs = {}
        for input_name, input_node in self.inputs.items():
            self._inputs[input_name] = np.zeros(parse_partial_shape(input_node[input_name].get_partial_shape()))


class ColorizationCheckModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info['adapter'])
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def predict(self, identifiers, input_data):
        input_dict = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in input_dict.items()})
        raw_result = self.exec_network.infer(input_dict)
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_result, result

    def fit_to_input(self, input_data):
        constant_normalization = 255.
        input_data *= constant_normalization
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        return {self.input_blob: input_data}


class ColorizationCheckOVModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info['adapter'])
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def predict(self, identifiers, input_data):
        input_dict = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in input_dict.items()})
        raw_result, raw_outputs = self.infer(input_dict, raw_results=True)
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_outputs, result

    def fit_to_input(self, input_data):
        constant_normalization = 255.
        input_data *= constant_normalization
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        return {self.input_blob: input_data}
