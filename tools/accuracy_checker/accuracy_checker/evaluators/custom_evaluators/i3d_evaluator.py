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
import warnings
import numpy as np

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher import create_launcher
from ...data_readers import create_reader
from ...utils import extract_image_representations, contains_all, get_path
from ...progress_reporters import ProgressReporter
from ...logging import print_info
from ...preprocessor import Crop, Resize


class I3DEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, adapter, rgb_model, flow_model):
        self.dataset_config = dataset_config
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self.adapter = adapter
        self.rgb_model = rgb_model
        self.flow_model = flow_model
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        dataset_config = config['datasets']
        launcher_settings = config['launchers'][0]
        supported_frameworks = ['dlsdk']
        if not launcher_settings['framework'] in supported_frameworks:
            raise ConfigError('{} framework not supported'.format(launcher_settings['framework']))
        if 'device' not in launcher_settings:
            launcher_settings['device'] = 'CPU'
        launcher = create_launcher(launcher_settings, delayed_model_loading=True)
        adapter = create_adapter(launcher_settings['adapter'])
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
        return cls(dataset_config, launcher, adapter, rgb_model, flow_model)

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

    def process_dataset(
            self, subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            allow_pairwise_subset=False,
            **kwargs):

        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)
        self._create_subset(subset, num_images, allow_pairwise_subset)

        self._annotations, self._predictions = [], []

        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )

        compute_intermediate_metric_res = kwargs.get('intermediate_metrics_results', False)
        if compute_intermediate_metric_res:
            metric_interval = kwargs.get('metrics_interval', 1000)
            ignore_results_formatting = kwargs.get('ignore_results_formatting', False)

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

            if _progress_reporter:
                _progress_reporter.update(batch_id, len(batch_prediction))
                if compute_intermediate_metric_res and _progress_reporter.current % metric_interval == 0:
                    self.compute_metrics(
                        print_results=True, ignore_results_formatting=ignore_results_formatting
                    )

        if _progress_reporter:
            _progress_reporter.finish()

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []

        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions
        ):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting)

        return self._metrics_results

    def print_metrics_results(self, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting)
            return
        result_presenters = self.metric_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_results(metric_result, ignore_results_formatting)

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

    @property
    def dataset_size(self):
        return self.dataset.size

    def release(self):
        self.rgb_model.release()
        self.flow_model.release()
        self.launcher.release()

    def reset(self):
        if self.metric_executor:
            self.metric_executor.reset()
        if hasattr(self, '_annotations'):
            del self._annotations
            del self._predictions
        del self._metrics_results
        self._annotations = []
        self._predictions = []
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

    def _create_subset(self, subset=None, num_images=None, allow_pairwise=False):
        if self.dataset.batch is None:
            self.dataset.batch = 1
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise)


class BaseModel:
    def __init__(self, network_info, launcher, data_source, delayed_model_loading=False):
        self.input_blob = None
        self.output_blob = None
        self.with_prefix = False
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

    def predict(self, input_data):
        return self.exec_network.infer(inputs=input_data[0])

    def release(self):
        del self.network
        del self.exec_network

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.auto_model_search(network_info, self.net_type)
        if weights:
            self.network = launcher.read_network(str(model), str(weights))
            self.network.batch_size = 1
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.network = None
            launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

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
            print_info('\tshape {}\n'.format(input_info.shape))
        print_info('{} - Output info'.format(self.net_type))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(output_info.shape))

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.input_blob].input_data
            if has_info else self.exec_network.inputs[self.input_blob]
        )
        input_data = np.array(input_data)
        input_data = np.transpose(input_data, (3, 0, 1, 2))
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
