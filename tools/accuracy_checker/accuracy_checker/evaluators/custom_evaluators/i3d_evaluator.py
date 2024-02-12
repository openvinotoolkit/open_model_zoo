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
limitations under the License."
"""

import warnings
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseDLSDKModel, BaseOpenVINOModel, BaseCascadeModel
from ...adapters import create_adapter
from ...config import ConfigError
from ...data_readers import create_reader
from ...utils import extract_image_representations, contains_all, parse_partial_shape
from ...preprocessor import Crop, Resize


def create_model(model_config, launcher, data_source, launcher_model_mapping, suffix=None, delayed_model_loading=False):
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, data_source, suffix, delayed_model_loading)


class I3DEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, adapter, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.adapter = adapter
        self.model = model
        if self.adapter is not None:
            self.adapter_type = self.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, launcher_config = cls.get_dataset_and_launcher_info(config)
        adapter = create_adapter(launcher_config['adapter'])
        data_source = dataset_config[0].get('data_source', None)

        model = I3DCascadeModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            data_source, delayed_model_loading
        )

        adapter.output_blob = model.output_blob
        return cls(dataset_config, launcher, adapter, model, orig_config)

    @staticmethod
    def get_dataset_info(dataset):
        identifiers = dataset.identifiers
        annotation = [dataset.annotation_provider[idx] for idx in identifiers]

        return annotation, identifiers

    @staticmethod
    def combine_predictions(output_rgb, output_flow):
        output = {}
        for key_rgb, key_flow in zip(output_rgb.keys(), output_flow.keys()):
            data_rgb = np.asarray(output_rgb[key_rgb])
            data_flow = np.asarray(output_flow[key_flow])

            if data_rgb.shape != data_flow.shape:
                raise ValueError("Calculation of combined output is not possible. Outputs for rgb and flow models have "
                                 "different shapes. rgb model's output shape: {}. "
                                 "flow model's output shape: {}.".format(data_rgb.shape, data_flow.shape))

            result_data = (data_rgb + data_flow) / 2
            output[key_rgb] = result_data

        return output

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        annotation, identifiers = self.get_dataset_info(self.dataset)
        for batch_id, (batch_annotation, batch_identifiers) in enumerate(zip(annotation, identifiers)):
            batch_inputs_images = self.model.rgb_model.prepare_data(batch_identifiers)
            batch_inputs_flow = self.model.flow_model.prepare_data(batch_identifiers)

            extr_batch_inputs_images, _ = extract_image_representations([batch_inputs_images])
            extr_batch_inputs_flow, _ = extract_image_representations([batch_inputs_flow])

            batch_raw_prediction_rgb = self.model.rgb_model.predict(batch_identifiers, extr_batch_inputs_images)
            batch_raw_prediction_flow = self.model.flow_model.predict(batch_identifiers, extr_batch_inputs_flow)
            batch_raw_out = self.combine_predictions(batch_raw_prediction_rgb, batch_raw_prediction_flow)

            batch_prediction = self.adapter.process([batch_raw_out], identifiers, [{}])

            if self.metric_executor.need_store_predictions:
                self._annotations.extend([batch_annotation])
                self._predictions.extend(batch_prediction)

            if self.metric_executor:
                self.metric_executor.update_metrics_on_batch(
                    [batch_id], [batch_annotation], batch_prediction
                )
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


class I3DCascadeModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, data_source=None, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['flow', 'rgb']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('configuration for flow/rgb does not exist')
        self._flow_mapping = {
            'dlsdk': I3DFlowModel,
            'openvino': I3DFlowOVModel
        }
        self._rgb_mapping = {
            'dlsdk': I3DRGBModel,
            'openvino': I3DRGBOVModel
        }
        self.flow_model = create_model(network_info.get('flow', {}), launcher, data_source, self._flow_mapping, 'flow',
                                       delayed_model_loading)
        self.rgb_model = create_model(network_info.get('rgb', {}), launcher, data_source, self._rgb_mapping, 'rgb',
                                      delayed_model_loading)
        if self.rgb_model.output_blob != self.flow_model.output_blob:
            warnings.warn("Outputs for rgb and flow models have different names. "
                          "rgb model's output name: {}. flow model's output name: {}. Output name of rgb model "
                          "will be used in combined output".format(self.rgb_model.output_blob,
                                                                   self.flow_model.output_blob))
        self.output_blob = self.rgb_model.output_blob

        self._part_by_name = {'flow_network': self.flow_model, 'rgb_network': self.rgb_model}

    def predict(self, identifiers, input_data, encoder_callback=None):
        pass


class BaseI3DModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, data_source, suffix=None, delayed_model_loading=False):
        reader_config = network_info.get('reader', {})
        source_prefix = reader_config.get('source_prefix', '')
        reader_config.update({
            'data_source': data_source / source_prefix
        })
        self.reader = create_reader(reader_config)
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data):
        input_dict = input_data[0]
        if self.dynamic_inputs and not self.is_dynamic:
            self._reshape_input({k: v.shape for k, v in input_dict.items()})
        return self.exec_network.infer(inputs=input_dict)

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.input_blob].input_data
            if has_info else self.exec_network.inputs[self.input_blob]
        )
        input_data = np.array(input_data)
        input_data = np.transpose(input_data, (3, 0, 1, 2))
        if not self.dynamic_inputs:
            input_data = np.reshape(input_data, input_info.shape)
        return {self.input_blob: input_data}


class BaseI3DOVModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, data_source, suffix=None, delayed_model_loading=False):
        reader_config = network_info.get('reader', {})
        source_prefix = reader_config.get('source_prefix', '')
        reader_config.update({
            'data_source': data_source / source_prefix
        })
        self.reader = create_reader(reader_config)
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data):
        input_dict = input_data[0]
        if self.dynamic_inputs and not self.is_dynamic:
            self._reshape_input({k: v.shape for k, v in input_dict.items()})
        return self.infer(input_dict)

    def fit_to_input(self, input_data):
        input_data = np.array(input_data)
        input_data = np.transpose(input_data, (3, 0, 1, 2))
        if not self.dynamic_inputs:
            input_data = np.reshape(input_data, parse_partial_shape(self.inputs[self.input_blob].get_partial_shape()))
        return {self.input_blob: input_data}


class I3DRGBModel(BaseI3DModel):
    def prepare_data(self, data):
        image_data = data.values[0]
        prepared_data = self.reader(image_data)
        prepared_data = self.preprocessing(prepared_data)
        prepared_data.data = self.fit_to_input(prepared_data.data)
        return prepared_data

    @staticmethod
    def preprocessing(image):
        resizer_config = {'type': 'resize', 'size': 256, 'aspect_ratio_scale': 'fit_to_window'}
        resizer = Resize(resizer_config)
        image = resizer.process(image)
        for i, frame in enumerate(image.data):
            image.data[i] = Crop.process_data(frame, 224, 224, None, False, False, True, {})
        return image


class I3DRGBOVModel(I3DRGBModel, BaseI3DOVModel):
    pass


class I3DFlowModel(BaseI3DModel):
    def prepare_data(self, data):
        numpy_data = data.values[1]
        prepared_data = self.reader(numpy_data)
        prepared_data.data = self.fit_to_input(prepared_data.data)
        return prepared_data


class I3DFlowOVModel(I3DFlowModel, BaseI3DOVModel):
    pass
