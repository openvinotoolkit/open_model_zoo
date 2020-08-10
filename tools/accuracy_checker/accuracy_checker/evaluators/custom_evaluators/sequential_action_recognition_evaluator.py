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
from functools import partial
from collections import OrderedDict
import numpy as np

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher import create_launcher
from ...utils import contains_all, contains_any, extract_image_representations, read_pickle, get_path
from ...progress_reporters import ProgressReporter
from ...logging import print_info


class SequentialActionRecognitionEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, model):
        self.dataset_config = dataset_config
        self.preprocessing_executor = None
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self.model = model
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'device' not in launcher_config:
            launcher_config['device'] = 'CPU'

        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        model = SequentialModel(
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
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_inputs)
            encoder_callback = None
            if output_callback:
                encoder_callback = partial(output_callback,
                                           metrics_result=None,
                                           element_identifiers=batch_identifiers,
                                           dataset_indices=batch_input_ids)

            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_inputs_extr, encoder_callback=encoder_callback
            )
            metrics_result = None
            if self.metric_executor:
                metrics_result = self.metric_executor.update_metrics_on_batch(
                    batch_input_ids, batch_annotation, batch_prediction
                )
                if self.metric_executor.need_store_predictions:
                    self._annotations.extend(batch_annotation)
                    self._predictions.extend(batch_prediction)

            if output_callback:
                output_callback(
                    batch_raw_prediction[0],
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids
                )
            if _progress_reporter:
                _progress_reporter.update(batch_id, len(batch_prediction))

        if _progress_reporter:
            _progress_reporter.finish()

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

    def load_network(self, network=None):
        self.model.load_network(network, self.launcher)

    def load_network_from_ir(self, models_list):
        self.model.load_model(models_list, self.launcher)

    def get_network(self):
        return self.model.get_network()

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

    def predict(self, idenitifiers, input_data):
        raise NotImplementedError

    def release(self):
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


def create_encoder(model_config, launcher, delayed_model_loading=False):
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
    return model_class(model_config, launcher, delayed_model_loading)


def create_decoder(model_config, launcher, delayed_model_loading):
    launcher_model_mapping = {
        'dlsdk': DecoderDLSDKModel,
        'onnx_runtime': DecoderONNXModel,
        'opencv': DecoderOpenCVModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


class SequentialModel(BaseModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        if models_args and not delayed_model_loading:
            encoder = network_info.get('encoder', {})
            decoder = network_info.get('decoder', {})
            if not contains_any(encoder, ['model', 'onnx_model']) and models_args:
                encoder['model'] = models_args[0]
                encoder['_model_is_blob'] = is_blob
            if not contains_any(decoder, ['model', 'onnx_model']) and models_args:
                decoder['model'] = models_args[1 if len(models_args) > 1 else 0]
                decoder['_model_is_blob'] = is_blob
            network_info.update({'encoder': encoder, 'decoder': decoder})
        if not contains_all(network_info, ['encoder', 'decoder']) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder and decoder fields')
        self.num_processing_frames = network_info['decoder'].get('num_processing_frames', 16)
        self.processing_frames_buffer = []
        self.encoder = create_encoder(network_info['encoder'], launcher, delayed_model_loading)
        self.decoder = create_decoder(network_info['decoder'], launcher, delayed_model_loading)
        self.store_encoder_predictions = network_info['encoder'].get('store_predictions', False)
        self._encoder_predictions = [] if self.store_encoder_predictions else None
        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder}

    def predict(self, identifiers, input_data, encoder_callback=None):
        raw_outputs = []
        predictions = []
        if len(np.shape(input_data)) == 5:
            input_data = input_data[0]
        for data in input_data:
            encoder_prediction = self.encoder.predict(identifiers, [data])
            if encoder_callback:
                encoder_callback(encoder_prediction)
            self.processing_frames_buffer.append(encoder_prediction[self.encoder.output_blob])
            if self.store_encoder_predictions:
                self._encoder_predictions.append(encoder_prediction[self.encoder.output_blob])
            if len(self.processing_frames_buffer) == self.num_processing_frames:
                raw_output, prediction = self.decoder.predict(identifiers, [self.processing_frames_buffer])
                raw_outputs.append(raw_output)
                predictions.append(prediction)
                self.processing_frames_buffer = []

        return raw_outputs, predictions

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

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, launcher)

    def _add_raw_encoder_predictions(self, encoder_prediction):
        for key, output in encoder_prediction.items():
            if key not in self._raw_outs:
                self._raw_outs[key] = []
            self._raw_outs[key].append(output)

    def get_network(self):
        return [{'name': 'encoder', 'model': self.encoder.network}, {'name': 'decoder', 'model': self.decoder.network}]


class EncoderDLSDKModel(BaseModel):
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        self.input_blob, self.output_blob = None, None
        self.with_prefix = None
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def load_model(self, network_info, launcher, log=False):
        if 'onnx_model' in network_info:
            network_info.update(launcher.config)
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def predict(self, identifiers, input_data):
        return self.exec_network.infer(self.fit_to_input(input_data))

    def release(self):
        del self.exec_network

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info[self.input_blob].input_data
        else:
            input_info = self.exec_network.inputs[self.input_blob]
        input_data = input_data.reshape(input_info.shape)

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
                    model_list = list(model.glob('*.xml'))
                    blob_list = list(model.glob('*.blob'))
                    if not model_list:
                        model_list = blob_list
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]
            print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        if model.suffix == '.blob':
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        print_info('{} - Found weights: {}'.format(self.default_model_suffix, weights))
        return model, weights

    def load_network(self, network, launcher):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.input_blob is None:
                output_blob = next(iter(self.exec_network.outputs))
            else:
                output_blob = (
                    '_'.join([self.default_model_suffix, self.output_blob])
                    if with_prefix else self.output_blob.split(self.default_model_suffix + '_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix


class DecoderDLSDKModel(BaseModel):
    default_model_suffix = 'decoder'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        self.input_blob, self.output_blob = None, None
        self.adapter = create_adapter(network_info.get('adapter', 'classification'))
        self.num_processing_frames = network_info.get('num_processing_frames', 16)
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)
        self.with_prefix = False

    def predict(self, identifiers, input_data):
        raw_result = self.exec_network.infer(self.fit_to_input(input_data))
        result = self.adapter.process([raw_result], identifiers, [{}])

        return raw_result, result

    def release(self):
        del self.exec_network

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.input_blob].input_data
            if has_info else self.exec_network.inputs[self.input_blob]
        )
        input_data = np.reshape(input_data, input_info.shape)
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
                    model_list = list(model.glob('*.xml'))
                    blob_list = list(model.glob('*.blob'))
                if not model_list and is_blob is None:
                    model_list = blob_list
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]
            print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        if model.suffix == '.blob':
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        print_info('{} - Found weights: {}'.format(self.default_model_suffix, weights))
        return model, weights

    def load_model(self, network_info, launcher, log=False):
        if 'onnx_model' in network_info:
            network_info.update(launcher.config)
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.network = None
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def load_network(self, network, launcher):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.input_blob is None:
                output_blob = next(iter(self.exec_network.outputs))
            else:
                output_blob = (
                    '_'.join([self.default_model_suffix, self.output_blob])
                    if with_prefix else self.output_blob.split(self.default_model_suffix + '_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix
            self.adapter.output_blob = self.output_blob


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

        return model


class DummyEncoder(BaseModel):
    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        if 'predictions' not in network_info:
            raise ConfigError('predictions_file is not found')
        self._predictions = read_pickle(network_info['predictions'])
        self.iterator = 0

    def predict(self, identifiers, input_data):
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
