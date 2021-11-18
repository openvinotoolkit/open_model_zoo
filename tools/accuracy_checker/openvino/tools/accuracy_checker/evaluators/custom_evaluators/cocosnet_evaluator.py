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
limitations under the License.
"""

from collections import OrderedDict
from pathlib import Path
import numpy as np
import cv2

from .base_custom_evaluator import BaseCustomEvaluator
from ...adapters import create_adapter
from ...config import ConfigError
from ...data_readers import DataRepresentation
from ...launcher.input_feeder import PRECISION_TO_DTYPE
from ...logging import print_info
from ...preprocessor import PreprocessingExecutor
from ...representation import RawTensorPrediction, RawTensorAnnotation
from ...utils import extract_image_representations, contains_all, get_path


class CocosnetEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, preprocessor_mask, preprocessor_image, gan_model,
                 check_model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.preprocessor_mask = preprocessor_mask
        self.preprocessor_image = preprocessor_image
        self.test_model = gan_model
        self.check_model = check_model
        self._part_by_name = {
            'gan_network': self.test_model,
        }
        if self.check_model:
            self._part_by_name.update({'verification_network': self.check_model})
        if hasattr(self.test_model, 'adapter'):
            self.adapter_type = self.test_model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)

        preprocessor_mask = PreprocessingExecutor(dataset_config[0].get('preprocessing_mask'))
        preprocessor_image = PreprocessingExecutor(dataset_config[0].get('preprocessing_image'))
        network_info = config.get('network_info', {})
        cocosnet_network = network_info.get('cocosnet_network', {})
        verification_network = network_info.get('verification_network', {})

        if not delayed_model_loading:
            model_args = config.get('_models', [])
            models_is_blob = config.get('_model_is_blob')

            if 'model' not in cocosnet_network and model_args:
                cocosnet_network['model'] = model_args[0]
                cocosnet_network['_model_is_blob'] = models_is_blob
            if verification_network and 'model' not in verification_network and model_args:
                verification_network['model'] = model_args[1 if len(model_args) > 1 else 0]
                verification_network['_model_is_blob'] = models_is_blob
            network_info.update({
                'cocosnet_network': cocosnet_network,
                'verification_network': verification_network
            })
            if not contains_all(network_info, ['cocosnet_network']):
                raise ConfigError('configuration for cocosnet_network does not exist')

        gan_model = CocosnetModel(network_info.get('cocosnet_network', {}), launcher, delayed_model_loading)
        if verification_network:
            check_model = GanCheckModel(network_info.get('verification_network', {}), launcher, delayed_model_loading)
        else:
            check_model = None

        return cls(
            dataset_config, launcher, preprocessor_mask, preprocessor_image, gan_model, check_model, orig_config
        )

    def _preprocessing_for_batch_input(self, batch_annotation, batch_inputs):
        for i, _ in enumerate(batch_inputs):
            for index_of_input, _ in enumerate(batch_inputs[i].data):
                preprocessor = self.preprocessor_mask
                if index_of_input % 2:
                    preprocessor = self.preprocessor_image
                batch_inputs[i].data[index_of_input] = preprocessor.process(
                    images=[DataRepresentation(batch_inputs[i].data[index_of_input])],
                    batch_annotation=batch_annotation)[0].data
        return batch_inputs

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self._preprocessing_for_batch_input(batch_annotation, batch_inputs)
            extr_batch_inputs, _ = extract_image_representations(batch_inputs)
            batch_predictions, raw_predictions = self.test_model.predict(batch_identifiers, extr_batch_inputs)
            annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions)

            if self.metric_executor:
                metrics_result, _ = self.metric_executor.update_metrics_on_batch(
                    batch_input_ids, annotations, predictions
                )
                check_model_annotations = []
                check_model_predictions = []
                if self.check_model:
                    for index_of_metric in range(self.check_model.number_of_metrics):
                        check_model_annotations.extend(
                            self.check_model.predict(batch_identifiers, annotations, index_of_metric)
                        )
                        check_model_predictions.extend(
                            self.check_model.predict(batch_identifiers, predictions, index_of_metric)
                        )
                    batch_identifiers.extend(batch_identifiers)
                    check_model_annotations = [
                        RawTensorAnnotation(batch_identifier, item)
                        for batch_identifier, item in zip(batch_identifiers, check_model_annotations)]
                    check_model_predictions = [
                        RawTensorPrediction(batch_identifier, item)
                        for batch_identifier, item in zip(batch_identifiers, check_model_predictions)]
                if self.metric_executor.need_store_predictions:
                    self._annotations.extend(check_model_annotations)
                    self._predictions.extend(check_model_predictions)
            if output_callback:
                output_callback(raw_predictions, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_predictions), csv_file)

    def load_model(self, network_list):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, self.launcher)

    def load_network(self, network=None):
        for network_dict in network:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], self.launcher)

    def get_network(self):
        return [{'name': key, 'model': model.network} for key, model in self._part_by_name.items()]

    def load_network_from_ir(self, models_list):
        model_paths = next(iter(models_list))
        next(iter(self._part_by_name.values())).load_model(model_paths, self.launcher)

class BaseModel:
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.input_blob = None
        self.output_blob = None
        self.with_prefix = False
        self.launcher = launcher
        self.is_dynamic = False
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    @staticmethod
    def auto_model_search(network_info, net_type=""):
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

    @property
    def inputs(self):
        if self.network:
            return self.network.input_info if hasattr(self.network, 'input_info') else self.network.inputs
        return self.exec_network.input_info if hasattr(self.exec_network, 'input_info') else self.exec_network.inputs

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def release(self):
        del self.network
        del self.exec_network

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.auto_model_search(network_info, self.net_type)
        if weights:
            self.network = launcher.read_network(model, weights)
            self.load_network(self.network, launcher)
        else:
            self.network = None
            self.exec_network = launcher.ie_core.import_network(str(model))
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
        self.set_input_and_output()

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

    def set_input_and_output(self):
        pass

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
            print_info('\tshape: {}\n'.format(
                input_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))
            print_info('{} - Output info'.format(self.net_type))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(
                output_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))


class CocosnetModel(BaseModel):
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.net_type = "cocosnet_network"
        self.adapter = create_adapter(network_info.get('adapter'))
        super().__init__(network_info, launcher, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            inputs_data = OrderedDict([(name, data.input_data) for name, data in self.exec_network.input_info.items()])
        else:
            inputs_data = self.exec_network.inputs
        self.inputs_names = list(inputs_data.keys())
        if self.output_blob is None:
            self.output_blob = next(iter(self.exec_network.outputs))

        if self.adapter.output_blob is None:
            self.adapter.output_blob = self.output_blob

    def fit_to_input(self, input_data):
        inputs = {}
        for value, key in zip(input_data, self.inputs_names):
            value = np.expand_dims(value, 0)
            value = np.transpose(value, (0, 3, 1, 2))
            inputs[key] = value.astype(PRECISION_TO_DTYPE[self.inputs[key].precision])
        return inputs

    def predict(self, identifiers, input_data):
        results = []
        for current_input in input_data:
            data = self.fit_to_input(current_input)
            if not self.is_dynamic and self.dynamic_inputs:
                self.reshape_net({k: v.shape for k, v in data.items()})
            prediction = self.exec_network.infer(data)
            results.append(*self.adapter.process(prediction, identifiers, [{}]))
        return results, prediction


class GanCheckModel(BaseModel):
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.net_type = "verification_network"
        self.additional_layers = network_info.get('additional_layers')
        super().__init__(network_info, launcher, delayed_model_loading)

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.auto_model_search(network_info, self.net_type)
        if weights:
            self.network = launcher.read_network(model, weights)
            for layer in self.additional_layers:
                self.network.add_outputs(layer)
            self.load_network(self.network, launcher)
        else:
            self.network = None
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        self.input_blob = next(iter(input_info))
        self.input_shape = tuple(input_info[self.input_blob].input_data.shape)
        self.output_blob = list(self.exec_network.outputs.keys())
        self.number_of_metrics = len(self.output_blob)

    def fit_to_input(self, input_data):
        input_data = cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR)
        input_data = cv2.resize(input_data, dsize=self.input_shape[2:])
        input_data = np.expand_dims(input_data, 0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        return {self.input_blob: input_data}

    def predict(self, identifiers, input_data, index_of_key):
        results = []
        for data in input_data:
            input_dict = self.fit_to_input(data.value)
            if not self.is_dynamic and self.dynamic_inputs:
                self.reshape_net({k, v.shape} for k, v in input_dict.items())
            prediction = self.exec_network.infer(input_dict)
            results.append(np.squeeze(prediction[self.output_blob[index_of_key]]))
        return results
