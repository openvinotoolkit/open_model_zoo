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

import numpy as np
from .text_to_speech_evaluator import TextToSpeechEvaluator, TTSDLSDKModel
from .base_models import BaseCascadeModel, BaseONNXModel, create_model
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all


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


class SequentialModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, adapter_info, is_blob=None, delayed_model_loading=False):
        super().__init__(network_info, launcher)
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
        self._encoder_mapping = {
            'dlsdk': EncoderOpenVINOModel,
            'onnx_runtime': EncoderONNXModel,
        }
        self._decoder_mapping = {
            'dlsdk': DecoderOpenVINOModel,
            'onnx_runtime': DecoderONNXModel
        }
        self.encoder = create_model(network_info['encoder'], launcher, self._encoder_mapping, 'encoder',
                                    delayed_model_loading)
        self.decoder = create_model(network_info['decoder'], launcher, self._decoder_mapping, 'decoder',
                                    delayed_model_loading)
        self.adapter = create_adapter(adapter_info)
        self.adapter.output_blob = 'audio'

        self.with_prefix = False
        self._part_by_name = {
            'encoder': self.encoder,
            'decoder': self.decoder,
        }

    def predict(self, identifiers, input_data, input_meta={}, input_names=None, callback=None):
        assert len(identifiers) == 1
        encoder_output, feats, chunk_size = self.encoder.predict(identifiers, input_data[0])
        if callback:
            callback(encoder_output)

        cfeats = encoder_output[self.encoder.output]
        decoder_data = (cfeats, feats, chunk_size)
        out_blob = self.decoder.predict(identifiers, decoder_data, callback=callback)

        return {}, self.adapter.process(out_blob, identifiers, input_meta)

    def load_model(self, network_list, launcher):
        super().load_model(network_list, launcher)
        self.update_inputs_outputs_info()

    def load_network(self, network_list, launcher):
        super().load_network(network_list, launcher)
        self.update_inputs_outputs_info()

    def update_inputs_outputs_info(self):
        current_name = next(iter(self.encoder.inputs))
        with_prefix = current_name.startswith('encoder_')
        if with_prefix != self.with_prefix:
            self.encoder.update_inputs_outputs_info(with_prefix)
            self.decoder.update_inputs_outputs_info(with_prefix)

        self.with_prefix = with_prefix


class EncoderModel:
    def predict(self, identifiers, input_data):
        features = np.resize(input_data, (-1, self.nb_features))
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

    def update_inputs_outputs_info(self, with_prefix):
        self.feature_input = generate_name(self.default_model_suffix+'_', with_prefix, self.feature_input)
        self.periods_input = generate_name(self.default_model_suffix+'_', with_prefix, self.periods_input)
        self.output = generate_name(self.default_model_suffix+'_', with_prefix, self.output)


class EncoderOpenVINOModel(EncoderModel, TTSDLSDKModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.nb_features = network_info.get('nb_features')
        self.nb_used_features = network_info.get('nb_used_features')
        self.feature_input = network_info.get('feature_input')
        self.periods_input = network_info.get('periods_input')
        self.output = network_info.get('output')
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def infer(self, feed_dict):
        feature_layer_shape = self.inputs[self.feature_input]
        if self.feature_input in self.dynamic_inputs or feature_layer_shape != feed_dict[self.feature_input].shape:
            input_shapes = {in_name: value.shape for in_name, value in feed_dict.items()}
            self._reshape_input(input_shapes)
        return self.exec_network.infer(feed_dict)


class EncoderONNXModel(BaseONNXModel, EncoderModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.is_dynamic = False
        self.nb_features = network_info.get('nb_features')
        self.nb_used_features = network_info.get('nb_used_features')
        self.feature_input = network_info.get('feature_input')
        self.periods_input = network_info.get('periods_input')
        self.output = network_info.get('output')
        outputs = self.inference_session.get_outputs()
        self.output_names = [output.name for output in outputs]

    @property
    def inputs(self):
        inputs_info = self.inference_session.get_inputs()
        return {input_layer.name: input_layer.shape for input_layer in inputs_info}

    def infer(self, feed_dict):
        outs = self.inference_session.run(self.output_names, feed_dict)
        return dict(zip(self.output_names, outs))


class DecoderModel:
    def predict(self, identifiers, input_data, order=16, callback=None):
        cfeats, features, chunk_size = input_data
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


class DecoderONNXModel(BaseONNXModel, DecoderModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.is_dynamic = False
        self.frame_size = network_info.get('frame_size')
        self.nb_frames = 1
        self.nb_features = network_info.get('nb_features')
        self.rnn_units1 = network_info.get('rnn_units1')
        self.rnn_units2 = network_info.get('rnn_units2')
        self.input1 = network_info.get('input1')
        self.input2 = network_info.get('input2')
        self.rnn_input1 = network_info.get('rnn_input1')
        self.rnn_input2 = network_info.get('rnn_input2')
        self.rnn_output1 = network_info.get('rnn_output1')
        self.rnn_output2 = network_info.get('rnn_output2')
        self.output = network_info.get('output')
        outputs = self.inference_session.get_outputs()
        self.output_names = [output.name for output in outputs]

    @property
    def inputs(self):
        inputs_info = self.inference_session.get_inputs()
        return {input_layer.name: input_layer.shape for input_layer in inputs_info}

    def infer(self, feed_dict):
        outs = self.inference_session.run(self.output_names, feed_dict)
        return dict(zip(self.output_names, outs))


class DecoderOpenVINOModel(DecoderModel, TTSDLSDKModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.frame_size = network_info.get('frame_size')
        self.nb_frames = 1
        self.nb_features = network_info.get('nb_features')
        self.rnn_units1 = network_info.get('rnn_units1')
        self.rnn_units2 = network_info.get('rnn_units2')
        self.input1 = network_info.get('input1')
        self.input2 = network_info.get('input2')
        self.rnn_input1 = network_info.get('rnn_input1')
        self.rnn_input2 = network_info.get('rnn_input2')
        self.rnn_output1 = network_info.get('rnn_output1')
        self.rnn_output2 = network_info.get('rnn_output2')
        self.output = network_info.get('output')
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def infer(self, feed_dict):
        return self.exec_network.infer(feed_dict)


class LPCNetEvaluator(TextToSpeechEvaluator):
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        adapter_info = config['adapter']
        model = SequentialModel(
            config.get('network_info', {}), launcher, config.get('_models', []), adapter_info,
            config.get('_model_is_blob'), delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)
