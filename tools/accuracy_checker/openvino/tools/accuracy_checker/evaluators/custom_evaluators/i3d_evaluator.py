"""
Copyright (c) 2018-2021 Intel Corporation

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

from pathlib import Path
from collections import OrderedDict
import warnings
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from ...adapters import create_adapter
from ...config import ConfigError
from ...data_readers import create_reader
from ...utils import extract_image_representations, contains_all, get_path
from ...logging import print_info
from ...preprocessor import Crop, Resize


class I3DEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, adapter, rgb_model, flow_model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.adapter = adapter
        self.rgb_model = rgb_model
        self.flow_model = flow_model
        self._part_by_name = {
            'flow_network': self.flow_model,
            'rgb_network': self.rgb_model
        }
        self.adapter_type = self.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, launcher_config = cls.get_dataset_and_launcher_info(config)
        adapter = create_adapter(launcher_config['adapter'])
        network_info = config.get('network_info', {})
        data_source = dataset_config[0].get('data_source', None)
        if not delayed_model_loading:
            flow_network = network_info.get('flow', {})
            rgb_network = network_info.get('rgb', {})
            model_args = config.get('_models', [])
            models_is_blob = config.get('_model_is_blob')
            if 'model' not in flow_network and model_args:
                flow_network['model'] = model_args[0]
                flow_network['_model_is_blob'] = models_is_blob
            if 'model' not in rgb_network and model_args:
                rgb_network['model'] = model_args[1 if len(model_args) > 1 else 0]
                rgb_network['_model_is_blob'] = models_is_blob
            network_info.update({
                'flow': flow_network,
                'rgb': rgb_network
            })
            if not contains_all(network_info, ['flow', 'rgb']):
                raise ConfigError('configuration for flow/rgb does not exist')

        flow_model = I3DFlowModel(
            network_info.get('flow', {}), launcher, data_source, delayed_model_loading
        )
        rgb_model = I3DRGBModel(
            network_info.get('rgb', {}), launcher, data_source, delayed_model_loading
        )
        if rgb_model.output_blob != flow_model.output_blob:
            warnings.warn("Outputs for rgb and flow models have different names. "
                          "rgb model's output name: {}. flow model's output name: {}. Output name of rgb model "
                          "will be used in combined output".format(rgb_model.output_blob, flow_model.output_blob))
        adapter.output_blob = rgb_model.output_blob
        return cls(dataset_config, launcher, adapter, rgb_model, flow_model, orig_config)

    @staticmethod
    def get_dataset_info(dataset):
        annotation = dataset.annotation_reader.annotation
        identifiers = dataset.annotation_reader.identifiers

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
            batch_inputs_images = self.rgb_model.prepare_data(batch_identifiers)
            batch_inputs_flow = self.flow_model.prepare_data(batch_identifiers)

            extr_batch_inputs_images, _ = extract_image_representations([batch_inputs_images])
            extr_batch_inputs_flow, _ = extract_image_representations([batch_inputs_flow])

            batch_raw_prediction_rgb = self.rgb_model.predict(extr_batch_inputs_images)
            batch_raw_prediction_flow = self.flow_model.predict(extr_batch_inputs_flow)
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


class BaseModel:
    def __init__(self, network_info, launcher, data_source, delayed_model_loading=False):
        self.input_blob = None
        self.output_blob = None
        self.with_prefix = False
        self.launcher = launcher
        self.is_dynamic = False
        reader_config = network_info.get('reader', {})
        source_prefix = reader_config.get('source_prefix', '')
        reader_config.update({
            'data_source': data_source / source_prefix
        })
        self.reader = create_reader(reader_config)
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    @staticmethod
    def auto_model_search(network_info, net_type):
        model = Path(network_info['model'])
        is_blob = network_info.get('_model_is_blob')
        if model.is_dir():
            if is_blob:
                model_list = list(model.glob('*.blob'))
            else:
                model_list = list(model.glob('*.xml'))
                if not model_list and is_blob is None:
                    model_list = list(model.glob('*.blob'))
            if not model_list:
                raise ConfigError('Suitable model not found')
            if len(model_list) > 1:
                raise ConfigError('Several suitable models found')
            model = model_list[0]
        accepted_suffixes = ['.blob', '.xml']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(net_type, model))
        if model.suffix == '.blob':
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(net_type, weights))

        return model, weights

    def reshape_net(self, shape):
        if self.is_dynamic:
            return
        if hasattr(self, 'exec_network') and self.exec_network is not None:
            del self.exec_network
        self.network.reshape(shape)
        self.dynamic_inputs, self.partial_shapes = self.launcher.get_dynamic_inputs(self.network)
        if not self.is_dynamic and self.dynamic_inputs:
            return
        self.exec_network = self.launcher.load_network(self.network, self.launcher.device)

    def predict(self, input_data):
        input_dict = input_data[0]
        if self.dynamic_inputs and not self.is_dynamic:
            self.reshape_net({k: v.shape for k, v in input_dict.items()})
        return self.exec_network.infer(inputs=input_dict)

    def release(self):
        del self.network
        del self.exec_network

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.auto_model_search(network_info, self.net_type)
        if weights:
            self.network = launcher.read_network(str(model), str(weights))
            self.network.batch_size = 1
            self.load_network(self.network, launcher)
        else:
            self.network = None
            launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def load_network(self, network, launcher):
        self.network = network
        self.dynamic_inputs, self.partial_shapes = launcher.get_dynamic_inputs(self.network)
        if self.dynamic_inputs and launcher.dynamic_shapes_policy in ['dynamic', 'default']:
            try:
                self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
                self.is_dynamic = True
            except RuntimeError as e:
                if launcher.dynamic_shapes_policy == 'dynamic':
                    raise e
                self.is_dynamic = False
                self.exec_network = None
        if not self.dynamic_inputs:
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith('{}_'.format(self.net_type))
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.input_blob is None:
                output_blob = next(iter(self.exec_network.outputs))
            else:
                output_blob = (
                    '_'.join([self.net_type, self.output_blob])
                    if with_prefix else self.output_blob.split('{}_'.format(self.net_type))[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix

    def print_input_output_info(self):
        print_info('{} - Input info:'.format(self.net_type))
        has_info = hasattr(self.network if self.network is not None else self.exec_network, 'input_info')
        if self.network:
            if has_info:
                network_inputs = OrderedDict(
                    [(name, data.input_data) for name, data in self.network.input_info.items()]
                )
            else:
                network_inputs = self.network.inputs
            network_outputs = self.network.outputs
        else:
            if has_info:
                network_inputs = OrderedDict([
                    (name, data.input_data) for name, data in self.exec_network.input_info.items()
                ])
            else:
                network_inputs = self.exec_network.inputs
            network_outputs = self.exec_network.outputs
        for name, input_info in network_inputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(input_info.precision))
            print_info('\tshape: {}\n'.format(input_info.shape))
        print_info('{} - Output info'.format(self.net_type))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(
                output_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))

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

    def prepare_data(self, data):
        pass


class I3DRGBModel(BaseModel):
    def __init__(self, network_info, launcher, data_source, delayed_model_loading=False):
        self.net_type = 'rgb'
        super().__init__(network_info, launcher, data_source, delayed_model_loading)

    def prepare_data(self, data):
        image_data = data[0]
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


class I3DFlowModel(BaseModel):
    def __init__(self, network_info, launcher, data_source, delayed_model_loading=False):
        self.net_type = 'flow'
        super().__init__(network_info, launcher, data_source, delayed_model_loading)

    def prepare_data(self, data):
        numpy_data = data[1]
        prepared_data = self.reader(numpy_data)
        prepared_data.data = self.fit_to_input(prepared_data.data)
        return prepared_data
