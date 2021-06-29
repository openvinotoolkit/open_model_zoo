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
import numpy as np
from .text_to_speech_evaluator import TextToSpeechEvaluator, TTSDLSDKModel
from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher import create_launcher
from ...utils import contains_all
from ...logging import print_info


scale = 255.0/32768.0
scale_1 = 32768.0/255.0


def ulaw2lin(u):
    u = u - 128
    s = np.sign(u)
    u = np.abs(u)
    return s*scale_1*(np.exp(u/128.*np.log(256))-1)


def lin2ulaw(x):
    s = np.sign(x)
    x = np.abs(x)
    u = (s*(128*np.log(1+scale*x)/np.log(256)))
    u = np.clip(128 + np.round(u), 0, 255)
    return u.astype('int16')


def generate_name(prefix, with_prefix, layer_name):
    return prefix + layer_name if with_prefix else layer_name.split(prefix)[-1]


class SequentialModel:
    def __init__(self, network_info, launcher, models_args, is_blob=None, delayed_model_loading=False):
        if not delayed_model_loading:
            encoder = network_info.get('encoder', {})
            decoder = network_info.get('decoder', {})
            if 'model' not in encoder:
                encoder['model'] = models_args[0]
                encoder['_model_is_blob'] = is_blob
            if 'model' not in decoder:
                decoder['model'] = models_args[1 if len(models_args) > 1 else 0]
                decoder['_model_is_blob'] = is_blob
            network_info.update({
                'encoder': encoder,
                'decoder': decoder,
            })
            required_fields = ['encoder', 'decoder']
            if not contains_all(network_info, required_fields):
                raise ConfigError(
                    'network_info should contains: {} fields'.format(' ,'.join(required_fields))
                )
        self.encoder = create_encoder(network_info, launcher, delayed_model_loading)
        self.decoder = create_decoder(network_info, launcher, delayed_model_loading)
        self.adapter = create_adapter(network_info['adapter'])
        self.adapter.output_blob = 'audio'

        self.with_prefix = False
        self._part_by_name = {
            'encoder': self.encoder,
            'decoder': self.decoder,
        }

    def predict(self, identifiers, input_data, input_meta, input_names=None, callback=None):
        assert len(identifiers) == 1
        encoder_output, feats, chunk_size = self.encoder.predict(input_data[0])
        if callback:
            callback(encoder_output)

        cfeats = encoder_output[self.encoder.output]
        out_blob = self.decoder.predict(cfeats, feats, chunk_size, callback=callback)

        return {}, self.adapter.process(out_blob, identifiers, input_meta)

    def release(self):
        self.encoder.release()
        self.decoder.release()

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, launcher)
        self.update_inputs_outputs_info()

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)
        self.update_inputs_outputs_info()

    def get_network(self):
        return [
            {'name': 'encoder', 'model': self.encoder.get_network()},
            {'name': 'decoder', 'model': self.decoder.get_network()},
        ]

    def update_inputs_outputs_info(self):
        current_name = next(iter(self.encoder.inputs))
        with_prefix = current_name.startswith('encoder_')
        if with_prefix != self.with_prefix:
            self.encoder.update_inputs_outputs_info(with_prefix)
            self.decoder.update_inputs_outputs_info(with_prefix)

        self.with_prefix = with_prefix


class EncoderModel:
    def __init__(self, network_info, launcher, suffix, nb_features, nb_used_features, delayed_model_loading=False):
        self.network_info = network_info
        self.nb_features = nb_features
        self.nb_used_features = nb_used_features
        self.feature_input = network_info.get('feature_input')
        self.periods_input = network_info.get('periods_input')
        self.output = network_info.get('output')
        self.default_model_suffix = suffix
        self.launcher = launcher
        self.prepare_model(launcher, network_info, delayed_model_loading)

    def predict(self, features):
        features = np.resize(features, (-1, self.nb_features))
        feature_chunk_size = features.shape[0]
        nb_frames = 1
        features = np.reshape(features, (nb_frames, feature_chunk_size, self.nb_features))
        features[:, :, 18:36] = 0
        periods = (.1 + 50 * features[:, :, 36:37] + 100).astype('float32')
        outs = self.infer(
            {
                self.feature_input: features[:, :, :self.nb_used_features],
                self.periods_input: periods
            })
        return outs, features, feature_chunk_size

    def infer(self, feed_dict):
        raise NotImplementedError

    def prepare_model(self, launcher, network_info, delayed_model_loading):
        raise NotImplementedError

    def update_inputs_outputs_info(self, with_prefix):
        self.feature_input = generate_name(self.default_model_suffix+'_', with_prefix, self.feature_input)
        self.periods_input = generate_name(self.default_model_suffix+'_', with_prefix, self.periods_input)
        self.output = generate_name(self.default_model_suffix+'_', with_prefix, self.output)


class EncoderOpenVINOModel(EncoderModel, TTSDLSDKModel):
    def prepare_model(self, launcher, network_info, delayed_model_loading):
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def infer(self, feed_dict):
        feature_layer_shape = self.inputs[self.feature_input]
        if feature_layer_shape != feed_dict[self.feature_input].shape:
            input_shapes = {in_name: value.shape for in_name, value in feed_dict.items()}
            self._reshape_input(input_shapes)
        return self.exec_network.infer(feed_dict)

    def _reshape_input(self, input_shapes):
        del self.exec_network
        self.network.reshape(input_shapes)
        self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)


class BaseONNXModel:
    @property
    def inputs(self):
        inputs_info = self.inference_session.get_inputs()
        return {input_layer.name: input_layer.shape for input_layer in inputs_info}

    def infer(self, feed_dict):
        outs = self.inference_session.run(self.output_names, feed_dict)
        return dict(zip(self.output_names, outs))

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

    def prepare_model(self, launcher, network_info, delayed_model_loading=False):
        if not delayed_model_loading:
            model = self.automatic_model_search(network_info)
            self.inference_session = launcher.create_inference_session(str(model))
            outputs = self.inference_session.get_outputs()
            self.output_names = [output.name for output in outputs]

class EncoderONNXModel(BaseONNXModel, EncoderModel):
    pass


class DecoderModel:
    def __init__(self, network_info, launcher, suffix, frame_size, nb_features, delayed_model_loading=False):
        self.network_info = network_info
        self.default_model_suffix = suffix
        self.frame_size = frame_size
        self.nb_frames = 1
        self.nb_features = nb_features
        self.rnn_units1 = network_info.get('rnn_units1')
        self.rnn_units2 = network_info.get('rnn_units2')
        self.input1 = network_info.get('input1')
        self.input2 = network_info.get('input2')
        self.rnn_input1 = network_info.get('rnn_input1')
        self.rnn_input2 = network_info.get('rnn_input2')
        self.rnn_output1 = network_info.get('rnn_output1')
        self.rnn_output2 = network_info.get('rnn_output2')
        self.output = network_info.get('output')
        self.prepare_model(launcher, network_info, delayed_model_loading)

    def predict(self, cfeats, features, chunk_size, order=16, callback=None):
        coef = 0.85
        pcm_chunk_size = self.frame_size * chunk_size
        pcm = np.zeros((self.nb_frames * pcm_chunk_size + order + 2,), dtype='float32')
        fexc = np.zeros((1, 1, 3), dtype='float32') + 128
        state1 = np.zeros((1, self.rnn_units1), dtype='float32')
        state2 = np.zeros((1, self.rnn_units2), dtype='float32')
        skip = order + 1
        mem = []
        mem_ = 0
        for fr in range(chunk_size):
            f = chunk_size + fr
            a = features[0, fr, self.nb_features - order:]
            for i in range(skip, self.frame_size):
                pcm_start_index = min(f*self.frame_size + i - 1, len(pcm) - 1)
                pred = -sum(a*pcm[pcm_start_index: pcm_start_index - order:-1])
                fexc[0, 0, 1] = lin2ulaw(pred)
                outputs = self.infer({
                    self.input1: fexc,
                    self.input2: cfeats[:, fr:fr + 1, :],
                    self.rnn_input1: state1,
                    self.rnn_input2: state2
                })

                if callback is not None:
                    callback(outputs)
                p = outputs[self.output]
                state1 = outputs[self.rnn_output1]
                state2 = outputs[self.rnn_output2]
                # Lower the temperature for voiced frames to reduce noisiness
                p *= np.power(p, np.maximum(0, 1.5 * features[0, fr, 37] - .5))
                p = p / (1e-18 + np.sum(p))
                # Cut off the tail of the remaining distribution
                p = np.maximum(p - 0.002, 0).astype('float64')
                p = p / (1e-8 + np.sum(p))
                rng = np.random.default_rng(12345)
                fexc[0, 0, 2] = np.argmax(rng.multinomial(1, p[0, 0, :], 1))
                pcm[pcm_start_index] = pred + ulaw2lin(fexc[0, 0, 2])
                fexc[0, 0, 0] = lin2ulaw(pcm[pcm_start_index])
                mem.append(coef * mem_ + pcm[pcm_start_index])
                skip = 0
        audio = np.round(mem).astype('int16')

        return {'audio':  audio}

    def update_inputs_outputs_info(self, with_prefix):
        prefix = self.default_model_suffix + '_'
        self.input1 = generate_name(prefix, with_prefix, self.input1)
        self.input2 = generate_name(prefix, with_prefix, self.input2)
        self.rnn_input1 = generate_name(prefix, with_prefix, self.rnn_input1)
        self.rnn_input2 = generate_name(prefix, with_prefix, self.rnn_input2)
        self.output = generate_name(prefix, with_prefix, self.output)
        self.rnn_output1 = generate_name(prefix, with_prefix, self.rnn_output1)
        self.rnn_output2 = generate_name(prefix, with_prefix, self.rnn_output2)

    def infer(self, feed_dict):
        raise NotImplementedError

    def prepare_model(self, launcher, network_info, delayed_model_loading=False):
        raise NotImplementedError


class DecoderONNXModel(BaseONNXModel, DecoderModel):
    pass


class DecoderOpenVINOModel(DecoderModel, TTSDLSDKModel):
    def prepare_model(self, launcher, network_info, delayed_model_loading=False):
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def infer(self, feed_dict):
        return self.exec_network.infer(feed_dict)


def create_encoder(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': EncoderOpenVINOModel,
        'onnx_runtime': EncoderONNXModel,
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(
        model_config['encoder'], launcher, 'encoder', model_config['nb_features'], model_config['nb_used_features'],
        delayed_model_loading
    )


def create_decoder(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': DecoderOpenVINOModel,
        'onnx_runtime': DecoderONNXModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(
        model_config['decoder'], launcher, 'decoder', model_config['frame_size'],
        model_config['nb_features'], delayed_model_loading
    )


class LPCNetEvaluator(TextToSpeechEvaluator):
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
