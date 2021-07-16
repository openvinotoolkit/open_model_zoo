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

from pathlib import Path
from collections import OrderedDict
import numpy as np
from .sr_evaluator import SuperResolutionFeedbackEvaluator
from ...adapters import create_adapter
from ...launcher import create_launcher
from ...logging import print_info
from ...utils import contains_any, contains_all, generate_layer_name, get_path, extract_image_representations
from ...config import ConfigError


def create_model(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': OpenVINOFeedbackModel,
        'onnx_runtime': ONNXFeedbackModel,
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


class FeedbackModel:
    def __init__(self, network_info, launcher):
        self.feedback = None
        self._feedback_shape = None
        self.adapter = create_adapter(network_info.get('adapter', 'background_matting'))

    def set_feedback(self, feedback):
        if np.ndim(feedback) == 2:
            feedback = np.expand_dims(feedback, -1)
        if np.shape(feedback)[0] == 1:
            feedback = np.transpose(feedback, (1, 2, 0))
        if feedback.max() > 1:
            feedback = feedback.astype(np.float32) / 255
        self.feedback = feedback
        self._feedback_shape = feedback.shape

    def reset_state(self):
        if self._feedback_shape is None:
            self.feedback = None
        else:
            self.feedback = np.zeros(self._feedback_shape)

    def predict(self, input_data, identifiers):
        return self.infer(input_data, identifiers)

    def infer(self, data, identifiers):
        raise NotImplementedError

    def release(self):
        pass


class ONNXFeedbackModel(FeedbackModel):
    default_model_suffix = 'segnet_model'

    def __init__(self, network_info, launcher, *args, **kwargs):
        super().__init__(network_info, launcher)
        model = self.automatic_model_search(network_info)
        self.inference_session = launcher.create_inference_session(str(model))
        self.input_blob = next(iter(self.inference_session.get_inputs()))
        self.output_blob = next(iter(self.inference_session.get_outputs()))

    def infer(self, data, identifiers):
        raw_results = self.inference_session.run((self.output_blob.name,), self.fit_to_input(data))
        results = self.adapter.process([{self.output_blob.name: raw_results[0]}], identifiers, [{}])

        return {self.output_blob: raw_results[0]}, results[0]

    def fit_to_input(self, input_data):
        if self.feedback is None:
            h, w = input_data.shape[:2]
            self.feedback = np.zeros((h, w, 1), dtype=np.float32)
        return {
            self.input_blob.name: np.expand_dims(
                np.transpose(np.concatenate([input_data, self.feedback], -1), (2, 0, 1)), 0
            ).astype(np.float32)
        }

    def release(self):
        del self.inference_session

    def automatic_model_search(self, network_info):
        model = Path(network_info['model'])
        if model.is_dir():
            model_list = list(model.glob('*{}.onnx'.format(self.default_model_suffix)))
            if not model_list:
                model_list = list(model.glob('*.onnx'))
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]
        accepted_suffixes = ['.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        return model


class OpenVINOFeedbackModel(FeedbackModel):
    default_model_suffix = 'segnet_model'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        self.input_blob, self.output_blob = None, None
        self.with_prefix = None
        self.launcher = launcher

        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)
            self.adapter = create_adapter(network_info.get('adapter', 'background_matting'))

    def infer(self, data, identifiers):
        raw_result = self.exec_network.infer(self.fit_to_input(data))
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_result, result[0]

    def fit_to_input(self, input_data):
        if self.feedback is None:
            h, w = input_data.shape[:2]
            self.feedback = np.zeros((h, w, 1), dtype=np.float32)
        return {self.input_blob: np.expand_dims(
            np.transpose(np.concatenate([input_data, self.feedback], -1), (2, 0, 1)), 0
        )}

    def release(self):
        del self.exec_network
        del self.launcher

    def update_inputs_outputs_info(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix + '_')
        if self.input_blob is None:
            self.input_blob = input_blob
            self.output_blob = next(iter(self.exec_network.outputs))
        if with_prefix != self.with_prefix:
            self.input_blob = generate_layer_name(self.input_blob, self.default_model_suffix, with_prefix)
            self.output_blob = generate_layer_name(self.output_blob, self.default_model_suffix, with_prefix)
            self.adapter.output_blob = self.output_blob

        self.with_prefix = with_prefix

    def print_input_output_info(self):
        print_info('{} - Input info:'.format(self.default_model_suffix))
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
            print_info('\tshape {}\n'.format(input_info.shape))
        print_info('{} - Output info'.format(self.default_model_suffix))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(output_info.shape))

    def automatic_model_search(self, network_info):
        model = Path(network_info.get('segnet_model', network_info.get('model')))
        if model.is_dir():
            is_blob = network_info.get('_model_is_blob')
            if is_blob:
                model_list = list(model.glob('*{}.blob'.format(self.default_model_suffix)))
                if not model_list:
                    model_list = list(model.glob('*.blob'))
            else:
                model_list = list(model.glob('*{}.xml'.format(self.default_model_suffix)))
                blob_list = list(model.glob('*{}.blob'.format(self.default_model_suffix)))
                if not model_list and not blob_list:
                    model_list = list(model.glob('*.xml'))
                    blob_list = list(model.glob('*.blob'))
                    if not model_list:
                        model_list = blob_list
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]
        accepted_suffixes = ['.blob', '.xml']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        if model.suffix == '.blob':
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(self.default_model_suffix, weights))

        return model, weights

    def load_network(self, network, launcher):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)
        self.update_inputs_outputs_info()

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.update_inputs_outputs_info()
        if log:
            self.print_input_output_info()


class VideoBackgroundMatting(SuperResolutionFeedbackEvaluator):
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'device' not in launcher_config:
            launcher_config['device'] = 'CPU'

        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        model = SegnetModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def process_dataset(
            self, subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            calculate_metrics=True,
            **kwargs):
        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)

        self._annotations, self._predictions = [], []

        self._create_subset(subset, num_images, allow_pairwise_subset)
        metric_config = self.configure_intermediate_metrics_results(kwargs)
        (compute_intermediate_metric_res, metric_interval, ignore_results_formatting,
         ignore_metric_reference) = metric_config

        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )
        previous_video_id = ''
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            if previous_video_id != batch_identifiers[0].video_id:
                self.model.reset()
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_inputs)

            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_inputs_extr
            )
            self.model.set_feedback(batch_prediction[0].value)
            previous_video_id = batch_prediction[0].identifier.video_id
            annotation, prediction = self.postprocessor.process_batch(batch_annotation, batch_prediction)

            metrics_result = None
            if self.metric_executor and calculate_metrics:
                metrics_result, _ = self.metric_executor.update_metrics_on_batch(
                    batch_input_ids, annotation, prediction
                )
                if self.metric_executor.need_store_predictions:
                    self._annotations.extend(annotation)
                    self._predictions.extend(prediction)

            if output_callback:
                output_callback(
                    batch_raw_prediction[0],
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids
                )
            if _progress_reporter:
                _progress_reporter.update(batch_id, len(prediction))
                if compute_intermediate_metric_res and _progress_reporter.current % metric_interval == 0:
                    self.compute_metrics(
                        print_results=True, ignore_results_formatting=ignore_results_formatting,
                        ignore_metric_reference=ignore_metric_reference
                    )
                    self.write_results_to_csv(kwargs.get('csv_result'), ignore_results_formatting, metric_interval)

        if _progress_reporter:
            _progress_reporter.finish()



class SegnetModel:
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        if models_args and not delayed_model_loading:
            model = network_info.get('segnet_model', {})
            if not contains_any(model, ['model', 'onnx_model']) and models_args:
                model['segnet_model'] = models_args[0]
                model['_model_is_blob'] = is_blob
            network_info.update({'sr_model': model})
        if not contains_all(network_info, ['segnet_model']) and not delayed_model_loading:
            raise ConfigError('network_info should contain segnet_model field')
        self.model = create_model(network_info['segnet_model'], launcher, delayed_model_loading)
        self._part_by_name = {'segnet_model': self.model}

    def predict(self, identifiers, input_data):
        predictions, raw_outputs = [], []
        for data in input_data:
            output, prediction = self.model.predict(data, identifiers)
            raw_outputs.append(output)
            predictions.append(prediction)
        return raw_outputs, predictions

    def reset(self):
        self.model.reset_state()

    def release(self):
        self.model.release()

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(
                network_dict.get('segnet_model', network_dict.get('model')), launcher)
        self.update_inputs_outputs_info()

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict.get('name', 'segnet_model')].load_model(network_dict, launcher)
        self.update_inputs_outputs_info()

    def get_network(self):
        return [{'name': 'segnet_model', 'model': self.model.network}]

    def update_inputs_outputs_info(self):
        if hasattr(self.model, 'update_inputs_outputs_info'):
            self.model.update_inputs_outputs_info()

    def set_feedback(self, feedback):
        self.model.set_feedback(feedback)
