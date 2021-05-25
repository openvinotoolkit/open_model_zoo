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
import pickle
from functools import partial
from collections import OrderedDict
import numpy as np

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher import create_launcher
from ...utils import contains_all, contains_any, extract_image_representations, get_path
from ...progress_reporters import ProgressReporter
from ...logging import print_info


def generate_name(prefix, with_prefix, layer_name):
    return prefix + layer_name if with_prefix else layer_name.split(prefix)[-1]


class SuperResolutionFeedbackEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, model):
        self.dataset_config = dataset_config
        self.preprocessing_executor = None
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self.srmodel = model
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'device' not in launcher_config:
            launcher_config['device'] = 'CPU'

        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        model = SRFModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model)

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
        compute_intermediate_metric_res, metric_interval, ignore_results_formatting = metric_config

        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )
        self.srmodel.init_feedback(self.dataset.data_reader)
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            self.srmodel.fill_feedback(batch_inputs)
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_inputs)
            callback = None
            if callback:
                callback = partial(output_callback,
                                   metrics_result=None,
                                   element_identifiers=batch_identifiers,
                                   dataset_indices=batch_input_ids)

            batch_raw_prediction, batch_prediction = self.srmodel.predict(
                batch_identifiers, batch_inputs_extr, callback=callback
            )
            annotation, prediction = self.postprocessor.process_batch(batch_annotation, batch_prediction)
            self.srmodel.feedback(prediction)

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
                        print_results=True, ignore_results_formatting=ignore_results_formatting
                    )

        if _progress_reporter:
            _progress_reporter.finish()

        if self.srmodel.store_predictions:
            self.srmodel.save_predictions()

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []

        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting)

        return self._metrics_results

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(False, ignore_results_formatting)

        result_presenters = self.metric_executor.get_metric_presenters()
        extracted_results, extracted_meta = [], []
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            result, metadata = presenter.extract_result(metric_result)
            if isinstance(result, list):
                extracted_results.extend(result)
                extracted_meta.extend(metadata)
            else:
                extracted_results.append(result)
                extracted_meta.append(metadata)
            if print_results:
                presenter.write_result(metric_result, ignore_results_formatting)

        return extracted_results, extracted_meta

    def print_metrics_results(self, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting)
            return
        result_presenters = self.metric_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_result(metric_result, ignore_results_formatting)

    @property
    def dataset_size(self):
        return self.dataset.size

    def release(self):
        self.srmodel.release()
        self.launcher.release()

    def reset(self):
        if self.metric_executor:
            self.metric_executor.reset()
        if hasattr(self, '_annotations'):
            del self._annotations
            del self._predictions
            del self._input_ids
        del self._metrics_results
        self._annotations = []
        self._predictions = []
        self._input_ids = []
        self._metrics_results = []
        if self.dataset:
            self.dataset.reset(self.postprocessor.has_processors)

    @staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        model_name = config['name']
        dataset_config = module_specific_params['datasets'][0]
        launcher_config = module_specific_params['launchers'][0]
        return (
            model_name, launcher_config['framework'], launcher_config['device'], launcher_config.get('tags'),
            dataset_config['name']
        )

    def _create_subset(self, subset=None, num_images=None, allow_pairwise=False):
        if self.dataset.batch is None:
            self.dataset.batch = 1
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise)

    @staticmethod
    def configure_intermediate_metrics_results(config):
        compute_intermediate_metric_res = config.get('intermediate_metrics_results', False)
        metric_interval, ignore_results_formatting = None, None
        if compute_intermediate_metric_res:
            metric_interval = config.get('metrics_interval', 1000)
            ignore_results_formatting = config.get('ignore_results_formatting', False)
        return compute_intermediate_metric_res, metric_interval, ignore_results_formatting

    def load_network(self, network=None):
        self.srmodel.load_network(network, self.launcher)

    def load_network_from_ir(self, models_list):
        self.srmodel.load_model(models_list, self.launcher)

    def get_network(self):
        return self.srmodel.get_network()

    def get_metrics_attributes(self):
        if not self.metric_executor:
            return {}
        return self.metric_executor.get_metrics_attributes()

    def register_metric(self, metric_config):
        if isinstance(metric_config, str):
            self.metric_executor.register_metric({'type': metric_config})
        elif isinstance(metric_config, dict):
            self.metric_executor.register_metric(metric_config)
        else:
            raise ValueError('Unsupported metric configuration type {}'.format(type(metric_config)))

    def register_postprocessor(self, postprocessing_config):
        pass

    def register_dumped_annotations(self):
        pass

    def select_dataset(self, dataset_tag):
        if self.dataset is not None and isinstance(self.dataset_config, list):
            return
        dataset_attributes = create_dataset_attributes(self.dataset_config, dataset_tag)
        self.dataset, self.metric_executor, self.preprocessor, self.postprocessor = dataset_attributes

    @staticmethod
    def _create_progress_reporter(check_progress, dataset_size):
        pr_kwargs = {}
        if isinstance(check_progress, int) and not isinstance(check_progress, bool):
            pr_kwargs = {"print_interval": check_progress}

        return ProgressReporter.provide('print', dataset_size, **pr_kwargs)


class BaseModel:
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def release(self):
        pass


# pylint: disable=E0203
class BaseDLSDKModel:
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
        model = Path(network_info.get('srmodel', network_info.get('model')))
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

    def update_inputs_outputs_info(self):
        raise NotImplementedError

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


def create_model(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': ModelDLSDKModel,
        'tf': ModelTFModel,
    }
    framework = launcher.config['framework']
    if 'predictions' in model_config and not model_config.get('store_predictions', False):
        framework = 'dummy'
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


class SRFModel(BaseModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        if models_args and not delayed_model_loading:
            model = network_info.get('srmodel', {})
            if not contains_any(model, ['model', 'onnx_model']) and models_args:
                model['srmodel'] = models_args[0]
                model['_model_is_blob'] = is_blob
            network_info.update({'sr_model': model})
        if not contains_all(network_info, ['srmodel']) and not delayed_model_loading:
            raise ConfigError('network_info should contain srmodel field')
        self.srmodel = create_model(network_info['srmodel'], launcher, delayed_model_loading)
        self.feedback = self.srmodel.feedback
        self.init_feedback = self.srmodel.init_feedback
        self.fill_feedback = self.srmodel.fill_feedback
        self.store_predictions = network_info['srmodel'].get('store_predictions', False)
        self._predictions = [] if self.store_predictions else None
        self._part_by_name = {'srmodel': self.srmodel}
        self._raw_outs = OrderedDict()

    def predict(self, identifiers, input_data, callback=None):
        predictions, raw_outputs = [], []
        for data in input_data:
            output, prediction = self.srmodel.predict(identifiers, data)
            if self.store_predictions:
                self._predictions.append(prediction)
            raw_outputs.append(output)
            predictions.append(prediction)
        return raw_outputs, predictions

    def reset(self):
        self.processing_frames_buffer = []
        if self._predictions is not None:
            self._predictions = []

    def release(self):
        self.srmodel.release()

    def save_predictions(self):
        if self._predictions is not None:
            prediction_file = Path(self.network_info['srmodel'].get('predictions', 'model_predictions.pickle'))
            with prediction_file.open('wb') as file:
                pickle.dump(self._predictions, file)

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(
                network_dict.get('srmodel', network_dict.get('model')), launcher)
        self.update_inputs_outputs_info()

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict.get('name', 'srmodel')].load_model(network_dict, launcher)
        self.update_inputs_outputs_info()

    def _add_raw_predictions(self, prediction):
        for key, output in prediction.items():
            if key not in self._raw_outs:
                self._raw_outs[key] = []
            self._raw_outs[key].append(output)

    def get_network(self):
        return [{'name': 'srmodel', 'model': self.srmodel.network}]

    def update_inputs_outputs_info(self):
        if hasattr(self.srmodel, 'update_inputs_outputs_info'):
            self.srmodel.update_inputs_outputs_info()


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


class ModelDLSDKModel(BaseModel, BaseDLSDKModel, FeedbackMixin):
    default_model_suffix = 'srmodel'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        self.input_blob, self.output_blob = None, None
        self.with_prefix = None

        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

        self.adapter = create_adapter(network_info.get('adapter', 'super_resolution'))
        self.configure_feedback()

    def predict(self, identifiers, input_data):
        input_data = self.fit_to_input(input_data)
        raw_result = self.exec_network.infer(input_data)
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_result, result

    def release(self):
        del self.exec_network
        del self.launcher

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
            assert tuple(info.input_data.shape) == np.shape(data)
            fitted[name] = data

        return fitted

    def update_inputs_outputs_info(self):
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

class ModelTFModel(BaseModel, FeedbackMixin):
    default_model_suffix = 'srmodel'

    def __init__(self, network_info, launcher, *args, **kwargs):
        super().__init__(network_info, launcher)
        model = self.automatic_model_search(network_info)
        self.inference_session = launcher.create_inference_session(str(model))

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

    def release(self):
        del self.inference_session

    @staticmethod
    def automatic_model_search(network_info):
        model = Path(network_info['model'])
        return model
