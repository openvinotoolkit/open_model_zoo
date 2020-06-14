"""
Copyright (c) 2019 Intel Corporation

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
import cv2

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher import create_launcher
from ...utils import extract_image_representations, contains_all, get_path
from ...progress_reporters import ProgressReporter
from ...logging import print_info


class ColorizationEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, test_model, check_model):
        self.dataset_config = dataset_config
        self.preprocessing_executor = None
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self.test_model = test_model
        self.check_model = check_model
        self._metrics_results = []
        self._part_by_name = {
            'colorization_network': self.test_model,
            'verification_network': self.check_model
        }

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
        network_info = config.get('network_info', {})
        if not delayed_model_loading:
            colorization_network = network_info.get('colorization_network', {})
            verification_network = network_info.get('verification_network', {})
            model_args = config.get('_models', [])
            models_is_blob = config.get('_model_is_blob')
            if 'model' not in colorization_network and model_args:
                colorization_network['model'] = model_args[0]
                colorization_network['_model_is_blob'] = models_is_blob
            if 'model' not in verification_network and model_args:
                verification_network['model'] = model_args[1 if len(model_args) > 1 else 0]
                verification_network['_model_is_blob'] = models_is_blob
            network_info.update({
                'colorization_network': colorization_network,
                'verification_network': verification_network
            })
            if not contains_all(network_info, ['colorization_network', 'verification_network']):
                raise ConfigError('configuration for colorization_network/verification_network does not exist')

        test_model = ColorizationTestModel(
            network_info.get('colorization_network', {}), launcher, delayed_model_loading
        )
        check_model = ColorizationCheckModel(
            network_info.get('verification_network', {}), launcher, delayed_model_loading
        )
        return cls(dataset_config, launcher, test_model, check_model)

    def process_dataset(
            self, subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotgiation=False,
            **kwargs):

        self._annotations, self._predictions = [], []
        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise_subset)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise_subset)
        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            extr_batch_inputs, _ = extract_image_representations(batch_inputs)
            metrics_result = None
            batch_raw_prediction, batch_out = self.test_model.predict(batch_identifiers, extr_batch_inputs)
            if output_callback:
                output_callback(
                    batch_raw_prediction,
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids
                )
            batch_raw_prediction, batch_prediction = self.check_model.predict(batch_identifiers, batch_out)
            if self.metric_executor:
                metrics_result = self.metric_executor.update_metrics_on_batch(
                    batch_input_ids, batch_annotation, batch_prediction
                )
                if self.metric_executor.need_store_predictions:
                    self._annotations.extend(batch_annotation)
                    self._predictions.extend(batch_prediction)

            if output_callback:
                output_callback(
                    batch_raw_prediction,
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids
                )
            if _progress_reporter:
                _progress_reporter.update(batch_id, len(batch_prediction))

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

    def release(self):
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

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict, launcher)

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)

    def get_network(self):
        return [
            {'name': 'colorization_network', 'model': self.test_model.network},
            {'name': 'verification_network', 'model': self.check_model.network}
        ]

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
        self.input_blob = None
        self.output_blob = None
        self.with_prefix = False
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
            print_info('{} - Found model: {}'.format(net_type, model))
        if model.suffix == '.blob':
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        print_info('{} - Found weights: {}'.format(net_type, weights))

        return model, weights

    def predict(self, idenitifers, input_data):
        raise NotImplementedError

    def release(self):
        pass

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


class ColorizationTestModel(BaseModel):
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.net_type = 'colorization_network'
        super().__init__(network_info, launcher, delayed_model_loading)
        self.color_coeff = np.load(network_info['color_coeff'])

    @staticmethod
    def data_preparation(input_data):
        input_ = input_data[0].astype(np.float32)
        img_lab = cv2.cvtColor(input_, cv2.COLOR_RGB2Lab)
        img_l = np.copy(img_lab[:, :, 0])
        img_l_rs = np.copy(img_lab[:, :, 0])
        return img_l, img_l_rs

    def postprocessing(self, res, img_l, output_blob, img_size):
        update_res = (res[output_blob] * self.color_coeff.transpose()[:, :, np.newaxis, np.newaxis]).sum(1)

        out = update_res.transpose((1, 2, 0)).astype(np.float32)
        out = cv2.resize(out, img_size)
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], out), axis=2)
        new_result = [np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)]
        return new_result

    def predict(self, identifiers, input_data):
        img_l, img_l_rs = self.data_preparation(input_data)
        h_orig, w_orig = input_data[0].shape[:2]
        res = self.exec_network.infer(inputs={self.input_blob: [img_l_rs]})

        new_result = self.postprocessing(res, img_l, self.output_blob, (w_orig, h_orig))
        return res, np.array(new_result)

    def release(self):
        del self.network
        del self.exec_network

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.input_blob].input_data
            if has_info else self.exec_network.inputs[self.input_blob]
        )
        input_data = np.reshape(input_data, input_info.shape)
        return {self.input_blob: input_data}

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith('colorization_network_')
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.input_blob is None:
                output_blob = next(iter(self.exec_network.outputs))
            else:
                output_blob = (
                    '_'.join(['colorization_network', self.output_blob])
                    if with_prefix else self.output_blob.split('colorization_network_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix


class ColorizationCheckModel(BaseModel):
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.net_type = 'verification_network'
        self.adapter = create_adapter(network_info['adapter'])
        super().__init__(network_info, launcher, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def predict(self, identifiers, input_data):
        raw_result = self.exec_network.infer(self.fit_to_input(input_data))
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_result, result

    def release(self):
        del self.network
        del self.exec_network

    def fit_to_input(self, input_data):
        constant_normalization = 255.
        input_data *= constant_normalization
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        return {self.input_blob: input_data}

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith('verification_network_')
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.input_blob is None:
                output_blob = next(iter(self.exec_network.outputs))
            else:
                output_blob = (
                    '_'.join(['verification_network', self.output_blob])
                    if with_prefix else self.output_blob.split('verification_network_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix
            self.adapter.output_blob = output_blob
