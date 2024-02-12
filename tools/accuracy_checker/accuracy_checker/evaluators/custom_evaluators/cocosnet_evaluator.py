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

from collections import OrderedDict
import numpy as np
import cv2

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseDLSDKModel, BaseOpenVINOModel, BaseCascadeModel, create_model
from ...adapters import create_adapter
from ...config import ConfigError
from ...data_readers import DataRepresentation
from ...launcher.input_feeder import PRECISION_TO_DTYPE
from ...preprocessor import PreprocessingExecutor
from ...representation import RawTensorPrediction, RawTensorAnnotation
from ...utils import extract_image_representations, contains_all, parse_partial_shape


class CocosnetEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, preprocessor_mask, preprocessor_image, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.preprocessor_mask = preprocessor_mask
        self.preprocessor_image = preprocessor_image
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)

        preprocessor_mask = PreprocessingExecutor(dataset_config[0].get('preprocessing_mask'))
        preprocessor_image = PreprocessingExecutor(dataset_config[0].get('preprocessing_image'))
        model = CocosnetCascadeModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )

        return cls(dataset_config, launcher, preprocessor_mask, preprocessor_image, model, orig_config)

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
            batch_predictions, raw_predictions = self.model.test_model.predict(batch_identifiers, extr_batch_inputs)
            annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions)

            if self.metric_executor:
                metrics_result, _ = self.metric_executor.update_metrics_on_batch(
                    batch_input_ids, annotations, predictions
                )
                check_model_annotations = []
                check_model_predictions = []
                if self.model.check_model:
                    for index_of_metric in range(self.model.check_model.number_of_metrics):
                        check_model_annotations.extend(
                            self.model.check_model.predict(batch_identifiers, annotations, index_of_metric)
                        )
                        check_model_predictions.extend(
                            self.model.check_model.predict(batch_identifiers, predictions, index_of_metric)
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


class CocosnetCascadeModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        if models_args and not delayed_model_loading:
            cocosnet_network = network_info.get('cocosnet_network', {})
            verification_network = network_info.get('verification_network', {})
            if 'model' not in cocosnet_network and models_args:
                cocosnet_network['model'] = models_args[0]
                cocosnet_network['_model_is_blob'] = is_blob
            if verification_network and 'model' not in verification_network and models_args:
                verification_network['model'] = models_args[1 if len(models_args) > 1 else 0]
                verification_network['_model_is_blob'] = is_blob
            network_info.update({
                'cocosnet_network': cocosnet_network,
                'verification_network': verification_network
            })
        if not contains_all(network_info, ['cocosnet_network']) and not delayed_model_loading:
            raise ConfigError('network_info should contain cocosnet_network field')
        self._test_mapping = {
            'dlsdk': CocosnetModel,
            'openvino': CoCosNetModelOV
        }
        self._check_mapping = {
            'dlsdk': GanCheckModel,
            'openvino': GANCheckOVModel
        }
        self.test_model = create_model(network_info.get('cocosnet_network', {}), launcher, self._test_mapping,
                                       'cocosnet_network', delayed_model_loading)
        if network_info.get('verification_network'):
            self.check_model = create_model(network_info.get('verification_network', {}), launcher, self._check_mapping,
                                            'verification_network', delayed_model_loading)
        else:
            self.check_model = None
        self._part_by_name = {
            'gan_network': self.test_model,
        }
        if self.check_model:
            self._part_by_name.update({'verification_network': self.check_model})

    @property
    def adapter(self):
        return self.test_model.adapter

    def predict(self, identifiers, input_data, encoder_callback=None):
        pass


class CocosnetModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    @property
    def inputs(self):
        if self.network:
            return self.network.input_info if hasattr(self.network, 'input_info') else self.network.inputs
        return self.exec_network.input_info if hasattr(self.exec_network, 'input_info') else self.exec_network.inputs

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
        prediction = None
        for current_input in input_data:
            data = self.fit_to_input(current_input)
            if not self.is_dynamic and self.dynamic_inputs:
                self._reshape_input({k: v.shape for k, v in data.items()})
            prediction = self.exec_network.infer(data)
            results.append(*self.adapter.process(prediction, identifiers, [{}]))
        return results, prediction


class CoCosNetModelOV(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def set_input_and_output(self):
        self.inputs_names = list(self.inputs.keys())
        if 'seg_map' in self.inputs_names[1]:
            img_input = self.inputs_names[-1]
            self.inputs_names[2] = self.inputs_names[1]
            self.inputs_names[1] = img_input
        if self.output_blob is None:
            self.output_blob = next(iter(self.exec_network.outputs)).get_node().friendly_name

        if self.adapter.output_blob is None:
            self.adapter.output_blob = self.output_blob

    def fit_to_input(self, input_data):
        inputs = {}
        for value, key in zip(input_data, self.inputs_names):
            value = np.expand_dims(value, 0)
            value = np.transpose(value, (0, 3, 1, 2))
            inputs[key] = value.astype(PRECISION_TO_DTYPE[self.inputs[key].element_type.get_type_name()])
        return inputs

    def predict(self, identifiers, input_data):
        results = []
        prediction = None
        if self.infer_request is None:
            self.infer_request = self.exec_network.create_infer_request()
        for current_input in input_data:
            data = self.fit_to_input(current_input)
            if not self.is_dynamic and self.dynamic_inputs:
                self._reshape_input({k: v.shape for k, v in data.items()})
            prediction, raw_prediction = self.infer(data, raw_results=True)
            results.append(*self.adapter.process(prediction, identifiers, [{}]))
        return results, raw_prediction


class GanCheckModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.additional_layers = network_info.get('additional_layers')
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    @property
    def inputs(self):
        if self.network:
            return self.network.input_info if hasattr(self.network, 'input_info') else self.network.inputs
        return self.exec_network.input_info if hasattr(self.exec_network, 'input_info') else self.exec_network.inputs

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
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

    def predict(self, identifiers, input_data, index_of_key=None):
        results = []
        for data in input_data:
            input_dict = self.fit_to_input(data.value)
            if not self.is_dynamic and self.dynamic_inputs:
                self._reshape_input({k, v.shape} for k, v in input_dict.items())
            prediction = self.exec_network.infer(input_dict)
            results.append(np.squeeze(prediction[self.output_blob[index_of_key]]))
        return results


class GANCheckOVModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.additional_layers = network_info.get('additional_layers')
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
        if model.suffix != '.blob':
            self.network = launcher.read_network(model, weights)
            self.network.add_outputs(self.additional_layers)
            self.load_network(self.network, launcher)
        else:
            self.network = None
            self.exec_network = launcher.ie_core.import_model(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def set_input_and_output(self):
        self.input_blob = next(iter(self.inputs))
        self.input_shape = parse_partial_shape(self.inputs[self.input_blob].get_partial_shape())
        self.output_blob = [out.get_node().friendly_name for out in self.exec_network.outputs]
        self.number_of_metrics = len(self.output_blob)

    def fit_to_input(self, input_data):
        input_data = cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR)
        input_data = cv2.resize(input_data, dsize=self.input_shape[2:])
        input_data = np.expand_dims(input_data, 0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        return {self.input_blob: input_data}

    def predict(self, identifiers, input_data, index_of_key=None):
        results = []
        for data in input_data:
            input_dict = self.fit_to_input(data.value)
            if not self.is_dynamic and self.dynamic_inputs:
                self._reshape_input({k, v.shape} for k, v in input_dict.items())
            prediction = self.infer(input_dict)
            results.append(np.squeeze(prediction[self.output_blob[index_of_key]]))
        return results
