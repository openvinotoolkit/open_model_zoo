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

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseCascadeModel, BaseDLSDKModel, BaseTFModel, BaseOpenVINOModel, create_model
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, parse_partial_shape


class SuperResolutionFeedbackEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = SRFModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        self.model.init_feedback(self.dataset.data_reader)
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            self.model.fill_feedback(batch_inputs)
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_inputs)
            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_inputs_extr
            )
            annotation, prediction = self.postprocessor.process_batch(batch_annotation, batch_prediction)
            self.model.feedback(prediction)
            metrics_result = self._get_metrics_result(batch_input_ids, annotation, prediction, calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction[0], metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(prediction), csv_file)


class SRFModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['srmodel']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain srmodel field')
        self._model_mapping = {
            'dlsdk': ModelDLSDKModel,
            'openvino': ModelOVModel,
            'tf': ModelTFModel,
        }
        self.srmodel = create_model(network_info['srmodel'], launcher, self._model_mapping, 'srmodel',
                                    delayed_model_loading)
        self.feedback = self.srmodel.feedback
        self.init_feedback = self.srmodel.init_feedback
        self.fill_feedback = self.srmodel.fill_feedback
        self._part_by_name = {'srmodel': self.srmodel}
        self._raw_outs = OrderedDict()

    def predict(self, identifiers, input_data):
        predictions, raw_outputs = [], []
        for data in input_data:
            output, prediction = self.srmodel.predict(identifiers, data)
            raw_outputs.append(output)
            predictions.append(prediction)
        return raw_outputs, predictions

    def _add_raw_predictions(self, prediction):
        for key, output in prediction.items():
            if key not in self._raw_outs:
                self._raw_outs[key] = []
            self._raw_outs[key].append(output)


class FeedbackMixin:
    def configure_feedback(self):
        self._idx_to_name = {}
        self._name_to_idx = {}
        self._feedback_name = self.network_info['feedback_input']
        self._feedback_data = {self._feedback_name: None}
        self._first_step = True
        self._inputs = self.network_info['inputs']
        self._feedback_inputs = {self._feedback_name: [t for t in self._inputs if t['name'] == self._feedback_name][0]}
        for input_info in self._inputs:
            idx = int(input_info['value'])
            self._idx_to_name[idx] = input_info['name']
            self._name_to_idx[input_info['name']] = idx
        self._feedback_idx = self._name_to_idx[self._feedback_name]

    def init_feedback(self, reader):
        info = self._feedback_inputs[self._feedback_name]
        self._feedback_data[self._feedback_name] = reader.read(info['initializer'])

    def feedback(self, data):
        data = data[0]
        self._feedback_data[self._feedback_name] = data[0].value

    def fill_feedback(self, data):
        data[0].data[self._feedback_idx] = self._feedback_data[self._feedback_name]
        return data


class ModelDLSDKModel(BaseDLSDKModel, FeedbackMixin):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter', 'super_resolution'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.partial_shapes = {}
        self.configure_feedback()

    def predict(self, identifiers, input_data):
        input_data = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: data.shape for key, data in input_data.items()})
        raw_result = self.exec_network.infer(input_data)
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_result, result

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info
        else:
            input_info = self.exec_network.inputs

        fitted = {}
        for name, info in input_info.items():
            data = input_data[self._name_to_idx[name]]
            data = np.expand_dims(data, axis=0)
            data = np.transpose(data, [0, 3, 1, 2])
            if not info.input_data.is_dynamic:
                assert tuple(info.input_data.shape) == np.shape(data)
            fitted[name] = data

        return fitted

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix + '_')
        if (with_prefix != self.with_prefix) and with_prefix:
            self.network_info['feedback_input'] = '_'.join([self.default_model_suffix,
                                                            self.network_info['feedback_input']])
            for inp in self.network_info['inputs']:
                inp['name'] = '_'.join([self.default_model_suffix, inp['name']])
                if 'blob' in inp.keys():
                    inp['blob'] = '_'.join([self.default_model_suffix, inp['blob']])
            self.network_info['adapter']['target_out'] = '_'.join([self.default_model_suffix,
                                                                   self.network_info['adapter']['target_out']])

        self.with_prefix = with_prefix

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        self.set_input_and_output()


class ModelOVModel(BaseOpenVINOModel, FeedbackMixin):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter', 'super_resolution'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.partial_shapes = {}
        self.configure_feedback()

    def predict(self, identifiers, input_data):
        input_data = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: data.shape for key, data in input_data.items()})
        raw_result, raw_t_results = self.infer(input_data, raw_results=True)
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_t_results, result

    def fit_to_input(self, input_data):
        fitted = {}
        for name, info in self.inputs.items():
            data = input_data[self._name_to_idx[name]]
            data = np.expand_dims(data, axis=0)
            if parse_partial_shape(info.get_partial_shape())[1] == 3:
                data = np.transpose(data, (0, 3, 1, 2))
            if not info.get_partial_shape().is_dynamic:
                assert tuple(parse_partial_shape(info.get_partial_shape())) == np.shape(data)
            fitted[name] = data

        return fitted

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        self.set_input_and_output()

    def set_input_and_output(self):
        input_info = self.inputs
        input_blob = next(iter(input_info))
        out_mapping = {}
        outputs = self.network.outputs if self.network is not None else self.exec_network.outputs
        for out in outputs:
            if not out.names:
                continue
            for name in out.names:
                out_mapping[name] = out.get_node().friendly_name
        self.adapter.additional_output_mapping = out_mapping
        with_prefix = input_blob.startswith(self.default_model_suffix + '_')
        if (with_prefix != self.with_prefix) and with_prefix:
            self.network_info['feedback_input'] = '_'.join([self.default_model_suffix,
                                                            self.network_info['feedback_input']])
            for inp in self.network_info['inputs']:
                inp['name'] = '_'.join([self.default_model_suffix, inp['name']])
                if 'blob' in inp.keys():
                    inp['blob'] = '_'.join([self.default_model_suffix, inp['blob']])

        self.with_prefix = with_prefix


class ModelTFModel(BaseTFModel, FeedbackMixin):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter = create_adapter(network_info.get('adapter', 'super_resolution'))
        self.configure_feedback()

    def predict(self, identifiers, input_data):
        input_data = self.fit_to_input(input_data)
        raw_result = self.inference_session.predict([input_data])
        result = self.adapter.process(raw_result, identifiers, [{}])
        return raw_result, result

    def fit_to_input(self, input_data):
        fitted = {}
        for idx, data in enumerate(input_data):
            name = self._idx_to_name[idx]
            data = np.expand_dims(data, axis=0)
            fitted[name] = data

        return fitted
