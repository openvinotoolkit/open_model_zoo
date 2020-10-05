"""
Copyright (c) 2018-2020 Intel Corporation

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
import copy
from collections import OrderedDict
import numpy as np
import cv2

from ..base_evaluator import BaseEvaluator
from ...utils import get_path, extract_image_representations
from ...dataset import Dataset
from ...launcher import create_launcher, InputFeeder, DummyLauncher
from ...launcher.loaders import PickleLoader
from ...logging import print_info
from ...metrics import MetricsExecutor
from ...postprocessor import PostprocessingExecutor
from ...preprocessor import PreprocessingExecutor
from ...adapters import create_adapter
from ...config import ConfigError
from ...data_readers import BaseReader, REQUIRES_ANNOTATIONS, DataRepresentation
from ...representation import RawTensorPrediction, RawTensorAnnotation


class CocosnetEvaluator(BaseEvaluator):
    def __init__(
            self, launcher, reader, preprocessor_mask, preprocessor_image, postprocessor,
            dataset, metric, model, model_for_metric
    ):
        self.launcher = launcher
        self.reader = reader
        self.preprocessor_mask = preprocessor_mask
        self.preprocessor_image = preprocessor_image
        self.postprocessor = postprocessor
        self.dataset = dataset
        self.metric_executor = metric
        self.model = model
        self.model_for_metric = model_for_metric
        self._annotations = []
        self._predictions = []
        self._metrics_results = []

    @classmethod
    def from_configs(cls, model_config):
        network_info = model_config['network_info']
        launcher_config = model_config['launchers'][0]
        dataset_config = model_config['datasets'][0]
        dataset_name = dataset_config['name']
        data_reader_config = dataset_config.get('reader', 'opencv_imread')
        data_source = dataset_config.get('data_source')
        dataset = Dataset(dataset_config)
        if isinstance(data_reader_config, str):
            data_reader_type = data_reader_config
            data_reader_config = None
        elif isinstance(data_reader_config, dict):
            data_reader_type = data_reader_config['type']
        else:
            raise ConfigError('reader should be dict or string')
        if data_reader_type in REQUIRES_ANNOTATIONS:
            data_source = dataset.annotation
        data_reader = BaseReader.provide(data_reader_type, data_source, data_reader_config)
        enable_ie_preprocessing = (
            dataset_config.get('_ie_preprocessing', False)
            if launcher_config['framework'] == 'dlsdk' else False
        )
        preprocessor_mask = PreprocessingExecutor(
            dataset_config.get('preprocessing_mask'), dataset_name, dataset.metadata,
            enable_ie_preprocessing=enable_ie_preprocessing
        )
        preprocessor_image = PreprocessingExecutor(
            dataset_config.get('preprocessing_image'), dataset_name, dataset.metadata,
            enable_ie_preprocessing=enable_ie_preprocessing
        )
        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        model = CocosnetModel(network_info, launcher)
        model_for_metric = GanCheckModel(network_info.get('verification_network', {}), launcher)
        postprocessor = PostprocessingExecutor(dataset_config.get('postprocessing'), dataset_name, dataset.metadata)
        metric_dispatcher = MetricsExecutor(dataset_config.get('metrics', []), dataset)

        return cls(
            launcher, data_reader,
            preprocessor_mask, preprocessor_image, postprocessor, dataset, metric_dispatcher, model, model_for_metric
        )

    @staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        launcher_config = module_specific_params['launchers'][0]
        dataset_config = module_specific_params['datasets'][0]

        return (
            config['name'],
            launcher_config['framework'], launcher_config['device'], launcher_config.get('tags'),
            dataset_config['name']
        )

    def _get_batch_input(self, batch_annotation):
        batch_identifiers = [annotation.identifier for annotation in batch_annotation]
        batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
        for annotation, input_data in zip(batch_annotation, batch_input):
            self.dataset.set_annotation_metadata(annotation, input_data, self.reader.data_source)
        for i, _ in enumerate(batch_input):
            for index_of_input, _ in enumerate(batch_input[i].data):
                preprocessor = self.preprocessor_mask
                if index_of_input % 2:
                    preprocessor = self.preprocessor_image
                batch_input[i].data[index_of_input] = preprocessor.process(
                    images=[DataRepresentation(batch_input[i].data[index_of_input])],
                    batch_annotation=batch_annotation)[0].data
        _, batch_meta = extract_image_representations(batch_input)

        return batch_input, batch_meta, batch_identifiers

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        if progress_reporter:
            progress_reporter.reset(self.dataset.size)

        if self.dataset.batch is None:
            self.dataset.batch = 1

        output_callback = kwargs.get('output_callback')
        predictions_to_store = []
        for batch_id, (batch_input_ids, batch_annotation) in enumerate(self.dataset):
            filled_inputs, batch_meta, batch_identifiers = self._get_batch_input(batch_annotation)
            batch_predictions = self.model.predict(batch_identifiers, batch_meta, filled_inputs)
            if stored_predictions:
                predictions_to_store.extend(copy.deepcopy(batch_predictions))
            annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions, batch_meta)
            self.metric_executor.update_metrics_on_batch(batch_input_ids, annotations, predictions)
            gan_annotations = []
            gan_predictions = []
            for index_of_metric in range(self.model_for_metric.number_of_metrics):
                gan_annotations.extend(self.model_for_metric.predict(batch_identifiers, batch_meta,
                                                                     annotations, index_of_metric))
                gan_predictions.extend(self.model_for_metric.predict(batch_identifiers, batch_meta,
                                                                     predictions, index_of_metric))
            batch_identifiers.extend(batch_identifiers)
            gan_annotations = [RawTensorAnnotation(batch_identifier, item)
                               for batch_identifier, item in zip(batch_identifiers, gan_annotations)]
            gan_predictions = [RawTensorPrediction(batch_identifier, item)
                               for batch_identifier, item in zip(batch_identifiers, gan_predictions)]

            if output_callback:
                output_callback(annotations, predictions)

            if self.metric_executor.need_store_predictions:
                self._annotations.extend(gan_annotations)
                self._predictions.extend(gan_predictions)
            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_predictions))

        if progress_reporter:
            progress_reporter.finish()

        if stored_predictions:
            self.store_predictions(stored_predictions, predictions_to_store)

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

    @property
    def metrics_results(self):
        if not self.metrics_results:
            self.compute_metrics(print_results=False)
        computed_metrics = copy.deepcopy(self._metrics_results)
        return computed_metrics

    @staticmethod
    def store_predictions(stored_predictions, predictions):
        # since at the first time file does not exist and then created we can not use it as a pathlib.Path object
        with open(stored_predictions, "wb") as content:
            pickle.dump(predictions, content)
            print_info("prediction objects are save to {}".format(stored_predictions))

    def reset_progress(self, progress_reporter):
        progress_reporter.reset(self.dataset.size)

    def reset(self):
        self.metric_executor.reset()
        del self._annotations
        del self._predictions
        del self._metrics_results
        self._annotations = []
        self._predictions = []
        self._metrics_results = []
        self.dataset.reset(self.postprocessor.has_processors)
        self.reader.reset()

    def release(self):
        self.model.release()
        self.launcher.release()


class BaseModel:
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.input_blob, self.output_blob = None, None
        self.with_prefix = None
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
            print_info('{} - Found model: {}'.format(net_type, model))
        if model.suffix == '.blob':
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        print_info('{} - Found weights: {}'.format(net_type, weights))

        return model, weights

    def predict(self, idenitifiers, input_data):
        raise NotImplementedError

    def release(self):
        pass

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.auto_model_search(network_info, self.net_type)
        if weights is not None:
            self.network = launcher.read_network(model, weights)
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def set_input_and_output(self):
        pass

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


class CorrespondenceNetwork(BaseModel):
    default_model_suffix = 'corr'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.net_type = "correspondence_network"
        super().__init__(network_info, launcher)
        self.input_feeder = InputFeeder(
            launcher.config.get('inputs', []), self.inputs
        )

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            self.inputs = OrderedDict([(name, data.input_data) for name, data in self.exec_network.input_info.items()])
        else:
            self.inputs = self.exec_network.inputs
        self.outputs = self.exec_network.outputs
        self.key_of_warped_reference = list(self.outputs.keys())[0]
        self.key_of_input_semantics = list(self.inputs.keys())[0]

    def fit_to_input(self, input_data):
        return self.input_feeder.fill_inputs(input_data)

    def inputs_info_for_meta(self):
        return {
            layer_name: layer.shape for layer_name, layer in self.inputs.items()
        }

    def release(self):
        del self.network
        del self.exec_network

    def predict(self, input_data):
        return self.exec_network.infer(input_data)


class GeneratorNetwork(BaseModel):
    default_model_suffix = 'gen'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.net_type = "generarative_network"
        super().__init__(network_info, launcher)
        self.adapter = create_adapter(network_info.get('adapter'))
        self.adapter.output_blob = self.output_blob

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        self.input_blob = next(iter(input_info))
        self.output_blob = next(iter(self.exec_network.outputs))

    def fit_to_input(self, input_data):
        return {self.input_blob: input_data}

    def release(self):
        del self.network
        del self.exec_network

    def predict(self, identifiers, meta, input_data):
        predictions = self.exec_network.infer(self.fit_to_input(input_data))
        result = self.adapter.process(predictions, identifiers, meta)
        return result


class CocosnetModel(BaseModel):
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.correspondence = CorrespondenceNetwork(network_info['correspondence'], launcher,
                                                    delayed_model_loading)
        self.generator = GeneratorNetwork(network_info['generator'], launcher, delayed_model_loading)

    def release(self):
        self.correspondence.release()
        self.generator.release()

    def predict(self, identifiers, meta, inputs):
        results = []
        for inp in inputs:
            inp = self.correspondence.fit_to_input([inp])
            corr_out = self.correspondence.predict(inp[0])
            gen_input = np.concatenate((corr_out[self.correspondence.key_of_warped_reference],
                                        inp[0][self.correspondence.key_of_input_semantics]), axis=1)
            result = self.generator.predict(identifiers, meta, gen_input)
            results.append(*result)
        return results


class GanCheckModel(BaseModel):
    default_model_suffix = 'check'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.net_type = "verification_network"
        self.additional_layers = network_info.get('additional_layers')
        super().__init__(network_info, launcher)

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.auto_model_search(network_info, self.net_type)
        if weights is not None:
            self.network = launcher.read_network(model, weights)
            for layer in self.additional_layers:
                self.network.add_outputs(layer)
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        self.input_blob = next(iter(input_info))
        self.output_blob = list(self.exec_network.outputs.keys())
        self.number_of_metrics = len(self.output_blob)

    def fit_to_input(self, input_data):
        input_data = cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR)
        input_data = cv2.resize(input_data, dsize=(299, 299))
        input_data = np.expand_dims(input_data, 0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        return {self.input_blob: input_data}

    def release(self):
        del self.network
        del self.exec_network

    def postprocessing(self, output):
        return np.squeeze(output)

    def predict(self, identifiers, meta, input_data, index_of_key):
        results = []
        for data in input_data:
            prediction = self.exec_network.infer(self.fit_to_input(data.value))
            results.append(self.postprocessing(prediction[self.output_blob[index_of_key]]))
        return results
