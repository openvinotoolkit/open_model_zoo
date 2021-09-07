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
from ...utils import contains_all, sigmoid, generate_layer_name


class Synthesizer:
    def __init__(self, network_info, launcher, models_args, is_blob=None, delayed_model_loading=False):
        if not delayed_model_loading:
            encoder = network_info.get('encoder', {})
            decoder = network_info.get('decoder', {})
            postnet = network_info.get('postnet', {})
            if 'model' not in encoder:
                encoder['model'] = models_args[0]
                encoder['_model_is_blob'] = is_blob
            if 'model' not in decoder:
                decoder['model'] = models_args[1 if len(models_args) > 1 else 0]
                decoder['_model_is_blob'] = is_blob
            if 'model' not in postnet:
                postnet['model'] = models_args[2 if len(models_args) > 2 else 0]
                postnet['_model_is_blob'] = is_blob

            network_info.update({
                'encoder': encoder,
                'decoder': decoder,
                'postnet': postnet
            })
            required_fields = ['encoder', 'decoder', 'postnet']
            if not contains_all(network_info, required_fields):
                raise ConfigError(
                    'network_info should contains: {} fields'.format(' ,'.join(required_fields))
                )
        self.encoder = create_encoder(network_info, launcher, delayed_model_loading)
        self.decoder = create_decoder(network_info, launcher, delayed_model_loading)
        self.postnet = create_postnet(network_info, launcher, delayed_model_loading)
        self.adapter = create_adapter(network_info['adapter'])

        self.with_prefix = False
        self._part_by_name = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'postnet': self.postnet
        }
        self.max_decoder_steps = int(network_info.get('max_decoder_steps', 500))
        self.gate_threshold = float(network_info.get('gate_treshold', 0.6))

    def predict(self, identifiers, input_data, input_meta, input_names=None, callback=None):
        assert len(identifiers) == 1
        encoder_outputs = self.encoder.predict(input_data[0])
        if callback:
            callback(encoder_outputs)
        postnet_outputs = []
        mel_outputs = []
        n = 0
        j = 0

        scheduler = [20] + [10] * 200
        offset = 20

        encoder_output = encoder_outputs[self.encoder.output_mapping['encoder_outputs']]
        feed_dict = self.decoder.init_feed_dict(encoder_output)
        for _ in range(self.max_decoder_steps):
            decoder_outs, feed_dict = self.decoder.predict(feed_dict)
            if callback:
                callback(decoder_outs)
            decoder_input = decoder_outs[self.decoder.output_mapping['decoder_input']]
            finished = decoder_outs[self.decoder.output_mapping['finished']]
            # padding for the first chunk for postnet
            if len(mel_outputs) == 0:
                mel_outputs = [decoder_input] * 10

            mel_outputs += [decoder_input]
            n += 1

            if n == scheduler[j]:
                postnet_input = np.transpose(np.array(mel_outputs[-scheduler[j] - offset:]), (1, 2, 0))
                postnet_outs = self.postnet.predict({self.postnet.input_mapping['mel_outputs']: postnet_input})
                if callback:
                    callback(postnet_outs)
                postnet_out = postnet_outs[self.postnet.output_mapping['postnet_outputs']]

                for k in range(postnet_out.shape[2]):
                    postnet_outputs.append(postnet_out[:, :, k])
                # yield here

                n = 0
                j += 1
            # process last chunk of frames, that might be shorter that scheduler
            if sigmoid(finished[0][0]) > self.gate_threshold:
                # right padding for the last chunk
                mel_outputs += [mel_outputs[-1]] * 10
                n += 10
                postnet_input = np.transpose(np.array(mel_outputs[-n - offset:]), (1, 2, 0))
                postnet_outs = self.postnet.predict({self.postnet.input_mapping['mel_outputs']: postnet_input})
                if callback:
                    callback(postnet_outs)
                postnet_out = postnet_outs[self.postnet.output_mapping['postnet_outputs']]

                for k in range(postnet_out.shape[2]):
                    postnet_outputs.append(postnet_out[:, :, k])
                break

        out_blob = {'postnet_outputs': np.array(postnet_outputs)[:, 0].reshape(1, -1, 22)}
        return {}, self.adapter.process(out_blob, identifiers, input_meta)

    def release(self):
        self.encoder.release()
        self.decoder.release()
        self.postnet.release()

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
            {'name': 'postnet', 'model': self.postnet.get_network()}
        ]

    def update_inputs_outputs_info(self):
        current_name = next(iter(self.encoder.inputs))
        with_prefix = current_name.startswith('encoder_')
        if with_prefix != self.with_prefix:
            self.encoder.update_inputs_outputs_info(with_prefix)
            self.decoder.update_inputs_outputs_info(with_prefix)
            self.postnet.update_inputs_outputs_info(with_prefix)

        self.with_prefix = with_prefix


class EncoderModel:
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.network_info = network_info
        self.input_mapping = {
            'text_encoder_outputs': 'text_encoder_outputs',
            'domain': 'domain',
            'f0s': 'f0s',
            'bert_embedding': 'bert_embedding'
        }
        self.output_mapping = {'encoder_outputs': 'encoder_outputs'}
        self.text_enc_dim = 384
        self.bert_dim = 768
        self.prepare_model(launcher, network_info, delayed_model_loading)
        self.launcher = launcher

    def predict(self, feed_dict):
        feed_dict = self.prepare_inputs(feed_dict)
        return self.infer(feed_dict)

    def infer(self, feed_dict):
        raise NotImplementedError

    def prepare_inputs(self, feed):
        feed[0] = feed[0].reshape(1, -1, self.text_enc_dim)
        feed[2] = feed[2].reshape(1, -1)
        feed[3] = feed[3].reshape(1, -1, self.bert_dim)
        return dict(zip(self.input_mapping.values(), feed))

    def prepare_model(self, launcher, network_info, delayed_model_loading):
        raise NotImplementedError

    def update_inputs_outputs_info(self, with_prefix):
        for input_id, input_name in self.input_mapping.items():
            self.input_mapping[input_id] = generate_layer_name(input_name, 'encoder_', with_prefix)

        for out_id, out_name in self.output_mapping.items():
            self.output_mapping[out_id] = generate_layer_name(out_name, 'encoder_', with_prefix)


class DecoderModel:
    default_model_suffix = 'decoder'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.network_info = network_info
        self.input_mapping = {
            'decoder_input': 'decoder_input',
            'attention_hidden': 'attention_hidden',
            'attention_cell': 'attention_cell',
            'decoder_hidden': 'decoder_hidden',
            'decoder_cell': 'decoder_cell',
            'attention_weights': 'attention_weights',
            'attention_weights_cum': 'attention_weights_cum',
            'attention_context': 'attention_context',
            'encoder_outputs': 'encoder_outputs'
        }
        self.output_mapping = {
            'finished': '109',
            'decoder_input': '108',
            'attention_hidden': '68',
            'attention_cell': '66',
            'decoder_hidden': '106',
            'decoder_cell': '104',
            'attention_weights': '85',
            'attention_weights_cum': '89',
            'attention_context': '88'
        }
        self.prepare_model(launcher, network_info, delayed_model_loading)
        self.launcher = launcher
        self.n_mel_channels = 22
        self.attention_rnn_dim = 800
        self.encoder_embedding_dim = 512
        self.decoder_rnn_dim = 800
        self.additional_inputs_filling = network_info.get('additional_input_filling', 'zeros')
        if self.additional_inputs_filling not in ['zeros', 'random']:
            raise ConfigError(
                'invalid setting for additional_inputs_filling: {}'.format(self.additional_inputs_filling)
            )
        self.seed = int(network_info.get('seed', 666))
        if self.additional_inputs_filling == 'random':
            np.random.seed(self.seed)

    @staticmethod
    def prepare_inputs(feed_dict):
        return feed_dict

    def predict(self, feed_dict):
        feed_dict = self.prepare_inputs(feed_dict)
        outputs = self.infer(feed_dict)
        return outputs, self.prepare_next_state_inputs(feed_dict, outputs)

    def infer(self, feed_dict):
        raise NotImplementedError

    def prepare_model(self, launcher, network_info, delayed_model_loading):
        raise NotImplementedError

    def prepare_next_state_inputs(self, feed_dict, outputs):
        common_layers = set(self.input_mapping).intersection(set(self.output_mapping))
        for common_layer in common_layers:
            feed_dict[self.input_mapping[common_layer]] = outputs[self.output_mapping[common_layer]]
        return feed_dict

    def update_inputs_outputs_info(self, with_prefix):
        for input_id, input_name in self.input_mapping.items():
            self.input_mapping[input_id] = generate_layer_name(input_name, 'decoder_', with_prefix)

        for out_id, out_name in self.output_mapping.items():
            self.output_mapping[out_id] = generate_layer_name(out_name, 'decoder_', with_prefix)

    def init_feed_dict(self, encoder_output):
        decoder_input = np.zeros((1, self.n_mel_channels), dtype=np.float32)
        attention_hidden = np.zeros((1, self.attention_rnn_dim), dtype=np.float32)
        attention_cell = np.zeros((1, self.attention_rnn_dim), dtype=np.float32)
        decoder_hidden = np.zeros((1, self.decoder_rnn_dim), dtype=np.float32)
        decoder_cell = np.zeros((1, self.decoder_rnn_dim), dtype=np.float32)
        attention_weights = np.zeros((1, encoder_output.shape[1]), dtype=np.float32)
        attention_weights_cum = np.zeros((1, encoder_output.shape[1]), dtype=np.float32)
        attention_context = np.zeros((1, self.encoder_embedding_dim), dtype=np.float32)
        return {
            self.input_mapping['decoder_input']: decoder_input,
            self.input_mapping['attention_hidden']: attention_hidden,
            self.input_mapping['attention_cell']: attention_cell,
            self.input_mapping['decoder_hidden']: decoder_hidden,
            self.input_mapping['decoder_cell']: decoder_cell,
            self.input_mapping['attention_weights']: attention_weights,
            self.input_mapping['attention_weights_cum']: attention_weights_cum,
            self.input_mapping['attention_context']: attention_context,
            self.input_mapping['encoder_outputs']: encoder_output
        }


class PostNetModel:
    default_model_suffix = 'postnet'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.network_info = network_info
        self.input_mapping = {'mel_outputs': 'mel_outputs'}
        self.output_mapping = {'postnet_outputs': 'postnet_outputs'}
        self.prepare_model(launcher, network_info, delayed_model_loading)

    def predict(self, feed_dict):
        feed_dict = self.prepare_inputs(feed_dict)
        return self.infer(feed_dict)

    @staticmethod
    def prepare_inputs(feed_dict):
        return feed_dict

    def infer(self, feed_dict):
        raise NotImplementedError

    def prepare_model(self, launcher, network_info, delayed_model_loading):
        raise NotImplementedError

    def update_inputs_outputs_info(self, with_prefix):
        for input_id, input_name in self.input_mapping.items():
            self.input_mapping[input_id] = generate_layer_name(input_name, 'postnet_', with_prefix)

        for out_id, out_name in self.output_mapping.items():
            self.output_mapping[out_id] = generate_layer_name(out_name, 'postnet_', with_prefix)


class EncoderOpenVINOModel(EncoderModel, TTSDLSDKModel):
    def prepare_model(self, launcher, network_info, delayed_model_loading):
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def prepare_inputs(self, feed):
        feed_dict = super().prepare_inputs(feed)
        if (
                feed_dict[self.input_mapping['text_encoder_outputs']].shape !=
                self.inputs[self.input_mapping['text_encoder_outputs']].input_data.shape
        ):
            new_shapes = {
                input_name: feed_dict[input_name].shape if input_name in feed_dict else self.inputs[input_name].shape
                for input_name in self.inputs
            }
            self._reshape_input(new_shapes)
        return feed_dict

    def infer(self, feed_dict):
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
        if hasattr(self, 'launcher'):
            self.launcher.release()

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

    def prepare_model(self, launcher, network_info, delayed_model_loading=False):
        if not delayed_model_loading:
            model = self.automatic_model_search(network_info)
            self.inference_session = launcher.create_inference_session(str(model))
            outputs = self.inference_session.get_outputs()
            self.output_names = [output.name for output in outputs]


class EncoderONNXModel(BaseONNXModel, EncoderModel):
    pass


class DecoderONNXModel(BaseONNXModel, DecoderModel):
    pass


class DecoderOpenVINOModel(DecoderModel, TTSDLSDKModel):
    def prepare_model(self, launcher, network_info, delayed_model_loading=False):
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def infer(self, feed_dict):
        return self.exec_network.infer(feed_dict)

    def prepare_inputs(self, feed_dict):
        if next(iter(self.input_mapping.values())) not in feed_dict:
            feed_dict_ = {self.input_mapping[input_name]: data for input_name, data in feed_dict.items()}
            feed_dict = feed_dict_
        if (
                feed_dict[self.input_mapping['encoder_outputs']].shape !=
                self.inputs[self.input_mapping['encoder_outputs']].input_data.shape
        ):
            new_shapes = {
                input_name: feed_dict[input_name].shape
                            if input_name in feed_dict else self.inputs[input_name].input_data.shape
                for input_name in self.inputs
            }
            self._reshape_input(new_shapes)

        if len(feed_dict) != len(self.inputs):
            extra_inputs = set(self.inputs).difference(set(feed_dict))
            for input_layer in extra_inputs:
                shape = self.inputs[input_layer].input_data.shape
                if self.additional_inputs_filling == 'zeros':
                    feed_dict[input_layer] = np.zeros(shape, dtype=np.float32)
                else:
                    feed_dict[input_layer] = np.random.uniform(size=shape)

        return feed_dict

    def _reshape_input(self, input_shapes):
        del self.exec_network
        self.network.reshape(input_shapes)
        self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)


class PostNetONNXModel(BaseONNXModel, PostNetModel):
    pass


class PostNetOpenVINOModel(PostNetModel, TTSDLSDKModel):
    def prepare_model(self, launcher, network_info, delayed_model_loading=False):
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def infer(self, feed_dict):
        return self.exec_network.infer(feed_dict)

    def prepare_inputs(self, feed_dict):
        if next(iter(self.input_mapping.values())) not in feed_dict:
            return {self.input_mapping[input_name]: data for input_name, data in feed_dict.items()}
        return feed_dict


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
        model_config['encoder'], launcher,
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
        model_config['decoder'], launcher, delayed_model_loading
    )


def create_postnet(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': PostNetOpenVINOModel,
        'onnx_runtime': PostNetONNXModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(
        model_config['postnet'], launcher, delayed_model_loading
    )


class Tacotron2Evaluator(TextToSpeechEvaluator):
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'device' not in launcher_config:
            launcher_config['device'] = 'CPU'

        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        model = Synthesizer(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model)
