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
import pickle
import numpy as np
from accuracy_checker.evaluators import BaseEvaluator
from accuracy_checker.dataset import Dataset
from accuracy_checker.adapters import create_adapter
from accuracy_checker.data_readers import BaseReader
from accuracy_checker.config import ConfigError
from accuracy_checker.preprocessor import PreprocessingExecutor
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.launcher import create_launcher
from accuracy_checker.utils import contains_all, contains_any, extract_image_representations, read_pickle


class SequentialActionRecognitionEvaluator(BaseEvaluator):
    def __init__(self, dataset, reader, preprocessing, metric_executor, launcher, model):
        self.dataset = dataset
        self.preprocessing_executor = preprocessing
        self.metric_executor = metric_executor
        self.launcher = launcher
        self.model = model
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
        launcher = create_launcher(config['launchers'][0], delayed_model_loading=True)
        model = SequentialModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob')
        )
        return cls(dataset, reader, preprocessing, metrics_executor, launcher, model)

    def process_dataset(self, stored_predictions, progress_reporter, *args, ** kwargs):
        self._annotations, self._predictions = ([], []) if self.metric_executor.need_store_predictions else None, None
        if progress_reporter:
            progress_reporter.reset(self.dataset.size)

        for batch_id, (dataset_indices, batch_annotation) in enumerate(self.dataset):
            batch_identifiers = [annotation.identifier for annotation in batch_annotation]
            batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
            batch_input = self.preprocessing_executor.process(batch_input, batch_annotation)
            batch_input, _ = extract_image_representations(batch_input)
            batch_prediction = self.model.predict(batch_identifiers, batch_input)
            self.metric_executor.update_metrics_on_batch(dataset_indices, batch_annotation, batch_prediction)
            if self.metric_executor.need_store_predictions:
                self._annotations.extend(batch_annotation)
                self._predictions.extend(batch_prediction)
            progress_reporter.update(batch_id, len(batch_prediction))

        if progress_reporter:
            progress_reporter.finish()

        if self.model.store_encoder_predictions:
            self.model.save_encoder_predictions()

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
        self.model.release()
        self.launcher.release()

    def reset(self):
        self.metric_executor.reset()
        self.model.reset()

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
    def __init__(self, network_info, launcher):
        self.network_info = network_info

    def predict(self, idenitifiers, input_data):
        raise NotImplementedError

    def release(self):
        pass


def create_encoder(model_config, launcher):
    launcher_model_mapping = {
        'dlsdk': EncoderDLSDKModel,
        'onnx_runtime': EncoderONNXModel,
        'opencv': EncoderOpenCVModel,
        'dummy': DummyEncoder
    }
    framework = launcher.config['framework']
    if 'predictions' in model_config and not model_config.get('store_predictions', False):
        framework = 'dummy'
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher)


def create_decoder(model_config, launcher):
    launcher_model_mapping = {
        'dlsdk': DecoderDLSDKModel,
        'onnx_runtime': DecoderONNXModel,
        'opencv': DecoderOpenCVModel,
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher)


class SequentialModel(BaseModel):
    def __init__(self, network_info, launcher, models_args, is_blob):
        super().__init__(network_info, launcher)
        if models_args:
            encoder = network_info.get('encoder', {})
            decoder = network_info.get('decoder', {})
            if not contains_any(encoder, ['model', 'onnx_model']) and models_args:
                encoder['model'] = models_args[0]
                encoder['_model_is_blob'] = is_blob
            if not contains_any(decoder, ['model', 'onnx_model']) and models_args:
                decoder['model'] = models_args[1 if len(models_args) > 1 else 0]
                decoder['_model_is_blob'] = is_blob
            network_info.update({'encoder': encoder, 'decoder': decoder})
        if not contains_all(network_info, ['encoder', 'decoder']):
            raise ConfigError('network_info should contains encoder and decoder fields')
        self.num_processing_frames = network_info['decoder'].get('num_processing_frames', 16)
        self.processing_frames_buffer = []
        self.encoder = create_encoder(network_info['encoder'], launcher)
        self.decoder = create_decoder(network_info['decoder'], launcher)
        self.store_encoder_predictions = network_info['encoder'].get('store_predictions', False)
        self._encoder_predictions = [] if self.store_encoder_predictions else None

    def predict(self, idenitifiers, input_data):
        predictions = []
        if len(np.shape(input_data)) == 5:
            input_data = input_data[0]
        for data in input_data:
            encoder_prediction = self.encoder.predict(idenitifiers, [data])
            self.processing_frames_buffer.append(encoder_prediction)
            if self.store_encoder_predictions:
                self._encoder_predictions.append(encoder_prediction)
            if len(self.processing_frames_buffer) == self.num_processing_frames:
                predictions.append(self.decoder.predict(idenitifiers, [self.processing_frames_buffer]))
                self.processing_frames_buffer = []

        return predictions

    def reset(self):
        self.processing_frames_buffer = []
        if self._encoder_predictions is not None:
            self._encoder_predictions = []

    def release(self):
        self.encoder.release()
        self.decoder.release()

    def save_encoder_predictions(self):
        if self._encoder_predictions is not None:
            prediction_file = Path(self.network_info['encoder'].get('predictions', 'encoder_predictions.pickle'))
            with prediction_file.open('wb') as file:
                pickle.dump(self._encoder_predictions, file)


class EncoderDLSDKModel(BaseModel):
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        if 'onnx_model' in network_info:
            network_info.update(launcher.config)
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            network = launcher.create_ie_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(network, launcher.device)
            self.input_blob = next(iter(network.inputs))
            self.output_blob = next(iter(network.outputs))
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
            self.input_blob = next(iter(self.exec_network.inputs))
            self.output_blob = next(iter(self.exec_network.outputs))

    def predict(self, identifiers, input_data):
        return self.exec_network.infer(self.fit_to_input(input_data))[self.output_blob]

    def release(self):
        del self.exec_network

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        input_data = input_data.reshape(self.exec_network.inputs[self.input_blob].shape)

        return {self.input_blob: input_data}

    def automatic_model_search(self, network_info):
        model = Path(network_info['model'])
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
                    model_list = list(model.glob('*.xml'.format(self.default_model_suffix)))
                    blob_list = list(model.glob('*.blob'.format(self.default_model_suffix)))
                    if not model_list:
                        model_list = blob_list
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]
        if model.suffix == '.blob':
            return model, None
        weights = network_info.get('weights', model.parent / model.name.replace('xml', 'bin'))

        return model, weights


class DecoderDLSDKModel(BaseModel):
    default_model_suffix = 'decoder'

    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        if 'onnx_model' in network_info:
            network_info.update(launcher.config)
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            network = launcher.create_ie_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(network, launcher.device)
            self.input_blob = next(iter(network.inputs))
            self.output_blob = next(iter(network.outputs))
        else:
            self.network = None
            self.exec_network = launcher.ie_core.import_network(str(model))
            self.input_blob = next(iter(self.exec_network.inputs))
            self.output_blob = next(iter(self.exec_network.outputs))

        self.adapter = create_adapter('classification')
        self.adapter.output_blob = self.output_blob
        self.num_processing_frames = network_info.get('num_processing_frames', 16)

    def predict(self, identifiers, input_data):
        result = self.exec_network.infer(self.fit_to_input(input_data))
        result = self.adapter.process([result], identifiers, [{}])

        return result

    def release(self):
        del self.exec_network

    def fit_to_input(self, input_data):
        input_data = np.reshape(input_data, self.exec_network.inputs[self.input_blob].shape)
        return {self.input_blob: input_data}

    def automatic_model_search(self, network_info):
        model = Path(network_info['model'])
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
                    model_list = list(model.glob('*.xml'.format(self.default_model_suffix)))
                    blob_list = list(model.glob('*.blob'.format(self.default_model_suffix)))
                if not model_list and is_blob is None:
                    model_list = blob_list
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]
        if model.suffix == '.blob':
            return model, None
        weights = network_info.get('weights', model.parent / model.name.replace('xml', 'bin'))
        return model, weights


class EncoderONNXModel(BaseModel):
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        model = self.automatic_model_search(network_info)
        self.inference_session = launcher.create_inference_session(str(model))
        self.input_blob = next(iter(self.inference_session.get_inputs()))
        self.output_blob = next(iter(self.inference_session.get_outputs()))

    def predict(self, identifiers, input_data):
        return self.inference_session.run((self.output_blob.name, ), self.fit_to_input(input_data))[0]

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        input_data = input_data.reshape(self.input_blob.shape)

        return {self.input_blob.name: input_data}

    def release(self):
        del self.inference_session

    def automatic_model_search(self, metwork_info):
        model = Path(metwork_info['model'])
        if model.is_dir():
            model_list = list(model.glob('*{}.onnx'.format(self.default_model_suffix)))
            if not model_list:
                model_list = list(model.glob('*.onnx'))
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]

        return model


class DecoderONNXModel(BaseModel):
    default_model_suffix = 'decoder'

    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        self.inference_session = launcher.create_inference_session(network_info['model'])
        self.input_blob = next(iter(self.inference_session.get_inputs()))
        self.output_blob = next(iter(self.inference_session.get_outputs()))
        self.adapter = create_adapter('classification')
        self.adapter.output_blob = self.output_blob.name
        self.num_processing_frames = network_info.get('num_processing_frames', 16)

    def predict(self, identifiers, input_data):
        result = self.inference_session.run((self.output_blob.name,), self.fit_to_input(input_data))
        return self.adapter.process([{self.output_blob.name: result[0]}], identifiers, [{}])

    def fit_to_input(self, input_data):
        input_data = np.reshape(input_data, self.input_blob.shape)
        return {self.input_blob.name: input_data}

    def release(self):
        del self.inference_session

    def automatic_model_search(self, metwork_info):
        model = Path(metwork_info['model'])
        if model.is_dir():
            model_list = list(model.glob('*{}.onnx'.format(self.default_model_suffix)))
            if not model_list:
                model_list = list(model.glob('*.onnx'))
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]

        return model


class DummyEncoder(BaseModel):
    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        if 'predictions' not in network_info:
            raise ConfigError('predictions_file is not found')
        self._predictions = read_pickle(network_info['predictions'])
        self.iterator = 0

    def predict(self, idenitifers, input_data):
        result = self._predictions[self.iterator]
        self.iterator += 1
        return result


class EncoderOpenCVModel(BaseModel):
    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        self.network = launcher.create_network(network_info['model'], network_info.get('weights', ''))
        network_info.update(launcher.config)
        input_shapes = launcher.get_inputs_from_config(network_info)
        self.input_blob = next(iter(input_shapes))
        self.input_shape = input_shapes[self.input_blob]
        self.network.setInputsNames(list(self.input_blob))
        self.output_blob = next(iter(self.network.getUnconnectedOutLayersNames()))

    def predict(self, identifiers, input_data):
        self.network.setInput(self.fit_to_input(input_data)[self.input_blob], self.input_blob)
        return self.network.forward([self.output_blob])[0]

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        input_data = input_data.reshape(self.input_shape)

        return {self.input_blob: input_data.astype(np.float32)}

    def release(self):
        del self.network


class DecoderOpenCVModel(BaseModel):
    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        self.network = launcher.create_network(network_info['model'], network_info.get('weights', ''))
        input_shapes = launcher.get_inputs_from_config(network_info)
        self.input_blob = next(iter(input_shapes))
        self.input_shape = input_shapes[self.input_blob]
        self.network.setInputsNames(list(self.input_blob))
        self.output_blob = next(iter(self.network.getUnconnectedOutLayersNames()))
        self.adapter = create_adapter('classification')
        self.adapter.output_blob = self.output_blob
        self.num_processing_frames = network_info.get('num_processing_frames', 16)

    def predict(self, identifiers, input_data):
        self.network.setInput(self.fit_to_input(input_data)[self.input_blob], self.input_blob)
        result = self.network.forward([self.output_blob])[0]
        return self.adapter.process([{self.output_blob.name: result}], identifiers, [{}])

    def fit_to_input(self, input_data):
        input_data = np.reshape(input_data, self.input_shape)
        return {self.input_blob: input_data.astype(np.float32)}

    def release(self):
        del self.network
