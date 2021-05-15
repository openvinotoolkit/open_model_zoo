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

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...adapters import create_adapter
from ...config import ConfigError
from ...data_readers import DataRepresentation
from ...launcher import create_launcher
from ...launcher.input_feeder import PRECISION_TO_DTYPE
from ...logging import print_info
from ...preprocessor import PreprocessingExecutor
from ...progress_reporters import ProgressReporter
from ...representation import RawTensorPrediction, RawTensorAnnotation
from ...utils import extract_image_representations, contains_all, get_path


class CocosnetEvaluator(BaseEvaluator):
    def __init__(
            self, dataset_config, launcher, preprocessor_mask, preprocessor_image,
            gan_model, check_model
    ):
        self.launcher = launcher
        self.dataset_config = dataset_config
        self.preprocessor_mask = preprocessor_mask
        self.preprocessor_image = preprocessor_image
        self.postprocessor = None
        self.dataset = None
        self.metric_executor = None
        self.test_model = gan_model
        self.check_model = check_model
        self._metrics_results = []
        self._part_by_name = {
            'gan_network': self.test_model,
        }
        if self.check_model:
            self._part_by_name.update({'verification_network': self.check_model})

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        launcher_config = config['launchers'][0]
        dataset_config = config['datasets']

        preprocessor_mask = PreprocessingExecutor(
            dataset_config[0].get('preprocessing_mask')
        )
        preprocessor_image = PreprocessingExecutor(
            dataset_config[0].get('preprocessing_image')
        )
        launcher = create_launcher(launcher_config, delayed_model_loading=True)

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
            dataset_config, launcher, preprocessor_mask, preprocessor_image, gan_model, check_model
        )

    @staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        model_name = config['name']
        launcher_config = module_specific_params['launchers'][0]
        dataset_config = module_specific_params['datasets'][0]

        return (
            model_name, launcher_config['framework'], launcher_config['device'], launcher_config.get('tags'),
            dataset_config['name']
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

    def process_dataset(
            self, subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            **kwargs):
        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)

        self._annotations, self._predictions = [], []

        self._create_subset(subset, num_images, allow_pairwise_subset)

        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )

        metric_config = self.configure_intermediate_metrics_results(kwargs)
        compute_intermediate_metric_res, metric_interval, ignore_results_formatting = metric_config

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
                output_callback(
                    raw_predictions,
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids
                )

            if _progress_reporter:
                _progress_reporter.update(batch_id, len(batch_predictions))
                if compute_intermediate_metric_res and _progress_reporter.current % metric_interval == 0:
                    self.compute_metrics(
                        print_results=True, ignore_results_formatting=ignore_results_formatting
                    )

        if _progress_reporter:
            _progress_reporter.finish()

        return self._annotations, self._predictions

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

    def release(self):
        self.test_model.release()
        if self.check_model:
            self.check_model.release()
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

    def load_model(self, network_list):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, self.launcher)

    def load_network(self, network_list):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], self.launcher)

    def get_network(self):
        return [{'name': key, 'model': model.network} for key, model in self._part_by_name.items()]

    def load_network_from_ir(self, models_list):
        model_paths = next(iter(models_list))
        next(iter(self._part_by_name.values())).load_model(model_paths, self.launcher)

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

    def set_profiling_dir(self, profiler_dir):
        self.metric_executor.set_profiling_dir(profiler_dir)

    def _create_subset(self, subset=None, num_images=None, allow_pairwise=False):
        if self.dataset.batch is None:
            self.dataset.batch = 1
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise)

    @staticmethod
    def _create_progress_reporter(check_progress, dataset_size):
        pr_kwargs = {}
        if isinstance(check_progress, int) and not isinstance(check_progress, bool):
            pr_kwargs = {"print_interval": check_progress}

        return ProgressReporter.provide('print', dataset_size, **pr_kwargs)

    @staticmethod
    def configure_intermediate_metrics_results(config):
        compute_intermediate_metric_res = config.get('intermediate_metrics_results', False)
        metric_interval, ignore_results_formatting = None, None
        if compute_intermediate_metric_res:
            metric_interval = config.get('metrics_interval', 1000)
            ignore_results_formatting = config.get('ignore_results_formatting', False)
        return compute_intermediate_metric_res, metric_interval, ignore_results_formatting

    @property
    def dataset_size(self):
        return self.dataset.size


class BaseModel:
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.input_blob = None
        self.output_blob = None
        self.with_prefix = False
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
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.network = None
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def load_network(self, network, launcher):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        self.set_input_and_output()

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
            print_info('\tshape {}\n'.format(input_info.shape))
        print_info('{} - Output info'.format(self.net_type))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(output_info.shape))


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

    def predict(self, identifiers, inputs):
        results = []
        for current_input in inputs:
            prediction = self.exec_network.infer(self.fit_to_input(current_input))
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
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
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
            prediction = self.exec_network.infer(self.fit_to_input(data.value))
            results.append(np.squeeze(prediction[self.output_blob[index_of_key]]))
        return results
