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
import numpy as np
import cv2

from accuracy_checker.evaluators.base_evaluator import BaseEvaluator
from accuracy_checker.dataset import Dataset
from accuracy_checker.adapters import create_adapter
from accuracy_checker.data_readers import BaseReader
from accuracy_checker.config import ConfigError
from accuracy_checker.preprocessor import PreprocessingExecutor
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.launcher import create_launcher
from accuracy_checker.utils import extract_image_representations, get_path, contains_all


class ColorizationEvaluator(BaseEvaluator):
    def __init__(self, dataset, reader, preprocessing, metric_executor, launcher, test_model, check_model):
        self.dataset = dataset
        self.preprocessing_executor = preprocessing
        self.metric_executor = metric_executor
        self.launcher = launcher
        self.test_model = test_model
        self.check_model = check_model
        self.reader = reader
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config):
        dataset_config = config['datasets'][0]
        dataset = Dataset(dataset_config)
        data_reader_config = dataset_config.get('reader', 'opencv_imread')
        data_source = dataset_config['data_source']
        if isinstance(data_reader_config, str):
            reader = BaseReader.provide(data_reader_config, data_source)
        elif isinstance(data_reader_config, dict):
            reader = BaseReader.provide(data_reader_config['type'], data_source, data_reader_config)
        else:
            raise ConfigError('reader should be dict or string')
        preprocessing = PreprocessingExecutor(dataset_config.get('preprocessing', []), dataset.name)
        metrics_executor = MetricsExecutor(dataset_config['metrics'], dataset)
        launcher_settings = config['launchers'][0]
        supported_frameworks = ['dlsdk']
        if not launcher_settings['framework'] in supported_frameworks:
            raise ConfigError('{} framework not supported'.format(launcher_settings['framework']))
        launcher = create_launcher(launcher_settings, delayed_model_loading=True)
        network_info = config.get('network_info', {})
        if not contains_all(network_info, ['colorization_network', 'verification_network']):
            raise ConfigError('colorization_network/verification_network does not exist')

        test_model = ColorizationTestModel(network_info['colorization_network'], launcher)
        check_model = ColorizationCheckModel(network_info['verification_network'], launcher)
        return cls(dataset, reader, preprocessing, metrics_executor, launcher, test_model, check_model)

    def process_dataset(self, _, progress_reporter):
        self._annotations, self._predictions = ([], []) if self.metric_executor.need_store_predictions else None, None
        if progress_reporter:
            progress_reporter.reset(self.dataset.size)

        for batch_id, (dataset_indices, batch_annotation) in enumerate(self.dataset):
            batch_identifiers = [annotation.identifier for annotation in batch_annotation]
            batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
            batch_input = self.preprocessing_executor.process(batch_input, batch_annotation)
            batch_input, _ = extract_image_representations(batch_input)
            batch_out = self.test_model.predict(batch_annotation, batch_input)
            batch_prediction = self.check_model.predict(batch_identifiers, batch_out)
            self.metric_executor.update_metrics_on_batch(dataset_indices, batch_annotation, batch_prediction)
            if self.metric_executor.need_store_predictions:
                self._annotations.extend(batch_annotation)
                self._predictions.extend(batch_prediction)
            progress_reporter.update(batch_id, len(batch_prediction))

        if progress_reporter:
            progress_reporter.finish()

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
        self.metric_executor.reset()

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


class BaseModel:
    def check_format(self, current_format):
        supported_format = ['.xml', '.bin']
        if not get_path(current_format).suffix in supported_format:
            raise ConfigError('{} format not supported'.format(supported_format))
        return current_format

    def predict(self, idenitifers, input_data):
        raise NotImplementedError

    def release(self):
        pass


class ColorizationTestModel(BaseModel):
    def __init__(self, network_info, launcher):
        super().__init__()
        model_xml = self.check_format(network_info['model'])
        model_bin = self.check_format(network_info['weights'])
        self.network = launcher.create_ie_network(model_xml, model_bin)
        self.network.batch_size = 1
        if not hasattr(launcher, 'plugin'):
            launcher.create_ie_plugin()
        self.exec_network = launcher.plugin.load(self.network)
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.test_mean = float(network_info['color_mean'])
        self.color_coeff = np.load(network_info['color_coeff'])

    def data_preparation(self, input_data):
        input = input_data[0].astype(np.float32)
        img_lab = cv2.cvtColor(input, cv2.COLOR_RGB2Lab)
        img_l = np.copy(img_lab[:, :, 0])
        img_l_rs = np.copy(img_lab[:, :, 0])
        img_l_rs -= self.test_mean
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

        output_blob = next(iter(self.exec_network.outputs))
        h_orig, w_orig = input_data[0].shape[:2]
        res = self.exec_network.infer(inputs={self.input_blob: [img_l_rs]})

        new_result = self.postprocessing(res, img_l, output_blob, (w_orig, h_orig))
        return np.array(new_result)

    def release(self):
        del self.exec_network

    def fit_to_input(self, input_data):
        input_data = np.reshape(input_data, self.network.inputs[self.input_blob].shape)
        return {self.input_blob: input_data}


class ColorizationCheckModel(BaseModel):
    def __init__(self, network_info, launcher):
        super().__init__()
        model_xml = self.check_format(network_info['model'])
        model_bin = self.check_format(network_info['weights'])
        self.adapter = create_adapter(network_info['adapter'])
        self.network = launcher.create_ie_network(model_xml, model_bin)
        if hasattr(launcher, 'plugin'):
            self.exec_network = launcher.plugin.load(self.network)
        else:
            launcher.load_network(self.network)
            self.exec_network = launcher.exec_network
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.adapter.output_blob = self.output_blob

    def predict(self, identifiers, input_data):
        result = self.exec_network.infer(self.fit_to_input(input_data))
        result = self.adapter.process([result], identifiers, [{}])
        return result

    def release(self):
        del self.exec_network

    def fit_to_input(self, input_data):
        constant_normalization = 255.
        input_data *= constant_normalization
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        return {self.input_blob: input_data}
