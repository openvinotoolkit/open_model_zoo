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
import pickle # nosec - disable B403:import-pickle check
from functools import partial
from collections import OrderedDict
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, contains_any, extract_image_representations, read_pickle, get_path
from ...logging import print_info


class AutomaticSpeechRecognitionEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = ASRModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_inputs)
            encoder_callback = None
            if output_callback:
                encoder_callback = partial(output_callback, metrics_result=None, element_identifiers=batch_identifiers,
                                           dataset_indices=batch_input_ids)
            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_inputs_extr, encoder_callback=encoder_callback
            )
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction[0], metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


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
    def _reshape_input(self, input_shapes):
        if not self.is_dynamic:
            del self.exec_network
            self.network.reshape(input_shapes)
            self.dynamic_inputs, self.partial_shapes = self.launcher.get_dynamic_inputs(self.network)
            if not self.is_dynamic and self.dynamic_inputs:
                self.exec_network = None
                return
            self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)

    def load_network(self, network, launcher):
        self.network = network
        self.dynamic_inputs, self.partial_shapes = launcher.get_dynamic_inputs(self.network)
        if self.dynamic_inputs and launcher.dynamic_shapes_policy in ['dynamic', 'default']:
            try:
                self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
                self.is_dynamic = True
            except RuntimeError as e:
                if launcher.dynamic_shapes_policy == 'dynamic':
                    raise e
                self.is_dynamic = False
                self.exec_network = None
                return
        if not self.dynamic_inputs:
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)

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
            print_info('\tshape {}\n'.format(
                input_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))
        print_info('{} - Output info'.format(self.default_model_suffix))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(
                output_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))

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

    def load_model(self, network_info, launcher, log=False):
        if 'onnx_model' in network_info:
            network_info.update(launcher.config)
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.load_network(self.network, launcher)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()


def create_encoder(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': EncoderDLSDKModel,
        'onnx_runtime': EncoderONNXModel,
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
        'onnx_runtime': DecoderONNXModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


class ASRModel(BaseModel):
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
        self._raw_outs = OrderedDict()

    def predict(self, identifiers, input_data, encoder_callback=None):
        predictions, raw_outputs = [], []
        for data in input_data:
            encoder_prediction, decoder_inputs = self.encoder.predict(identifiers, data)
            if encoder_callback:
                encoder_callback(encoder_prediction)
            if self.store_encoder_predictions:
                self._encoder_predictions.append(encoder_prediction)
            raw_output, prediction = self.decoder.predict(identifiers, decoder_inputs)
            raw_outputs.append(raw_output)
            predictions.append(prediction)
        return raw_outputs, predictions

    def reset(self):
        self.processing_frames_buffer = []
        if self._encoder_predictions is not None:
            self._encoder_predictions = []

    @property
    def adapter(self):
        return self.decoder.adapter

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


class EncoderDLSDKModel(BaseModel, BaseDLSDKModel):
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        self.input_blob, self.output_blob = None, None
        self.with_prefix = None
        self.is_dynamic = False
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def predict(self, identifiers, input_data):
        input_data = self.fit_to_input(input_data)
        results = self.exec_network.infer(input_data)
        return results, results[self.output_blob]

    def release(self):
        del self.exec_network
        del self.launcher

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info[self.input_blob].input_data
        else:
            input_info = self.exec_network.inputs[self.input_blob]
        if self.input_blob in self.dynamic_inputs or tuple(input_info.shape) != np.shape(input_data):
            self._reshape_input({self.input_blob: np.shape(input_data)})

        return {self.input_blob: np.array(input_data)}


class DecoderDLSDKModel(BaseModel, BaseDLSDKModel):
    default_model_suffix = 'decoder'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        self.input_blob, self.output_blob = None, None
        self.adapter = create_adapter(network_info.get('adapter', 'ctc_greedy_decoder'))
        self.is_dynamic = False
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)
        self.with_prefix = False

    def predict(self, identifiers, input_data):
        feed_dict = self.fit_to_input(input_data)
        raw_result = self.exec_network.infer(feed_dict)
        result = self.adapter.process([raw_result], identifiers, [{}])

        return raw_result, result

    def release(self):
        del self.exec_network
        del self.launcher

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.input_blob].input_data
            if has_info else self.exec_network.inputs[self.input_blob]
        )
        input_data = np.array(input_data)
        if tuple(input_info.shape) != input_data.shape:
            self._reshape_input({self.input_blob: input_data.shape})

        return {self.input_blob: input_data}

    def set_input_and_output(self):
        super().set_input_and_output()
        self.adapter.output_blob = self.output_blob


class EncoderONNXModel(BaseModel):
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher, *args, **kwargs):
        super().__init__(network_info, launcher)
        model = self.automatic_model_search(network_info)
        self.inference_session = launcher.create_inference_session(str(model))
        self.input_blob = next(iter(self.inference_session.get_inputs()))
        self.output_blob = next(iter(self.inference_session.get_outputs()))

    def predict(self, identifiers, input_data):
        results = self.inference_session.run((self.output_blob.name, ), self.fit_to_input(input_data))
        return results, results[0]

    def fit_to_input(self, input_data):
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
        accepted_suffixes = ['.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        return model


class DecoderONNXModel(BaseModel):
    default_model_suffix = 'decoder'

    def __init__(self, network_info, launcher, *args, **kwargs):
        super().__init__(network_info, launcher)
        self.inference_session = launcher.create_inference_session(network_info['model'])
        self.input_blob = next(iter(self.inference_session.get_inputs()))
        self.output_blob = next(iter(self.inference_session.get_outputs()))
        self.adapter = create_adapter(network_info['adapter'])
        self.adapter.output_blob = self.output_blob.name

    def predict(self, identifiers, input_data):
        result = self.inference_session.run((self.output_blob.name,), self.fit_to_input(input_data))
        return result, self.adapter.process([{self.output_blob.name: result[0]}], identifiers, [{}])

    def fit_to_input(self, input_data):
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
        accepted_suffixes = ['.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_suffix, model))

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
        return None, result
