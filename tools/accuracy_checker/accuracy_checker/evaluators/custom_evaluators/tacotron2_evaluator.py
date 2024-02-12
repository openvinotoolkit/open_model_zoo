"""
Copyright (c) 2018-2024 Intel Corporation

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
from .text_to_speech_evaluator import TextToSpeechEvaluator, TTSDLSDKModel, TTSOVModel
from .base_models import BaseCascadeModel, BaseONNXModel, create_model
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, sigmoid, generate_layer_name, parse_partial_shape, postprocess_output_name


class Synthesizer(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, adapter_info, is_blob=None, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['encoder', 'decoder', 'postnet']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder, decoder and postnet fields')
        self._encoder_mapping = {
            'dlsdk': EncoderDLSDKModel,
            'openvino': EncoderOpenVINOModel,
            'onnx_runtime': EncoderONNXModel,
        }
        self._decoder_mapping = {
            'dlsdk': DecodeDLSDKModel,
            'openvino': DecodeOpenVINOModel,
            'onnx_runtime': DecoderONNXModel
        }
        self._postnet_mapping = {
            'dlsdk': PostNetDLSDKModel,
            'openvino': PostNetOpenVINOModel,
            'onnx_runtime': PostNetONNXModel
        }
        self.encoder = create_model(network_info['encoder'], launcher, self._encoder_mapping, 'encoder',
                                    delayed_model_loading)
        self.decoder = create_model(network_info['decoder'], launcher, self._decoder_mapping, 'decoder',
                                    delayed_model_loading)
        self.postnet = create_model(network_info['postnet'], launcher, self._postnet_mapping, 'postnet',
                                    delayed_model_loading)
        self.adapter = create_adapter(adapter_info)

        self.with_prefix = False
        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder, 'postnet': self.postnet}
        self.max_decoder_steps = int(network_info.get('max_decoder_steps', 500))
        self.gate_threshold = float(network_info.get('gate_treshold', 0.6))

    def predict(self, identifiers, input_data, input_meta=None, input_names=None, callback=None):
        assert len(identifiers) == 1
        encoder_outputs = self.encoder.predict(identifiers, input_data[0])
        encoder_outputs = send_callback(encoder_outputs, callback)
        postnet_outputs = []
        mel_outputs = []
        n = 0
        j = 0

        scheduler = [20] + [10] * 200
        offset = 20

        encoder_output = encoder_outputs[self.encoder.output_mapping['encoder_outputs']]
        feed_dict = self.decoder.init_feed_dict(encoder_output)
        for _ in range(self.max_decoder_steps):
            decoder_outs, feed_dict = self.decoder.predict(identifiers, feed_dict)
            decoder_outs = send_callback(decoder_outs, callback)
            decoder_input = decoder_outs[self.decoder.output_mapping['decoder_input']]
            finished = decoder_outs[self.decoder.output_mapping['finished']]
            # padding for the first chunk for postnet
            if len(mel_outputs) == 0:
                mel_outputs = [decoder_input] * 10

            mel_outputs += [decoder_input]
            n += 1

            if n == scheduler[j]:
                postnet_input = np.transpose(np.array(mel_outputs[-scheduler[j] - offset:]), (1, 2, 0))
                postnet_outs = self.postnet.predict(identifiers,
                                                    {self.postnet.input_mapping['mel_outputs']: postnet_input})
                postnet_outs = send_callback(postnet_outs, callback)
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
                postnet_outs = self.postnet.predict(identifiers,
                                                    {self.postnet.input_mapping['mel_outputs']: postnet_input})
                postnet_outs = send_callback(postnet_outs, callback)
                postnet_out = postnet_outs[self.postnet.output_mapping['postnet_outputs']]

                for k in range(postnet_out.shape[2]):
                    postnet_outputs.append(postnet_out[:, :, k])
                break

        out_blob = {'postnet_outputs': np.array(postnet_outputs)[:, 0].reshape(1, -1, 22)}
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
            self.postnet.update_inputs_outputs_info(with_prefix)

        self.with_prefix = with_prefix


def send_callback(outs, callback):
    if isinstance(outs, tuple):
        outs, raw_outs = outs
    else:
        raw_outs = outs
    if callback:
        callback(raw_outs)
    return outs


class EncoderModel:
    def predict(self, identifiers, input_data):
        feed_dict = self.prepare_inputs(input_data)
        return self.infer(feed_dict)

    def prepare_inputs(self, feed):
        feed[0] = feed[0].reshape(1, -1, self.text_enc_dim)
        feed[2] = feed[2].reshape(1, -1)
        feed[3] = feed[3].reshape(1, -1, self.bert_dim)
        return dict(zip(self.input_mapping.values(), feed))

    def update_inputs_outputs_info(self, with_prefix):
        for input_id, input_name in self.input_mapping.items():
            self.input_mapping[input_id] = generate_layer_name(input_name, 'encoder_', with_prefix)
        if hasattr(self, 'outputs'):
            for out_id, out_name in self.output_mapping.items():
                o_name = postprocess_output_name(
                    out_name, self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False)
                if o_name not in self.outputs:
                    o_name = postprocess_output_name(
                    generate_layer_name(out_name, 'encoder_', with_prefix),
                    self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False)
                self.output_mapping[out_id] = o_name


class DecoderModel:
    def predict(self, identifiers, input_data):
        feed_dict = self.prepare_inputs(input_data)
        outputs = self.infer(feed_dict)
        if isinstance(outputs, tuple):
            return outputs, self.prepare_next_state_inputs(feed_dict, outputs)
        return outputs, self.prepare_next_state_inputs(feed_dict, outputs)

    def prepare_next_state_inputs(self, feed_dict, outputs):
        common_layers = set(self.input_mapping).intersection(set(self.output_mapping))
        if isinstance(outputs, tuple):
            outs = outputs[0]
        else:
            outs = outputs
        for common_layer in common_layers:
            feed_dict[self.input_mapping[common_layer]] = outs[self.output_mapping[common_layer]]
        return feed_dict

    def update_inputs_outputs_info(self, with_prefix):
        for input_id, input_name in self.input_mapping.items():
            self.input_mapping[input_id] = generate_layer_name(input_name, 'decoder_', with_prefix)
        if hasattr(self, 'outputs'):
            for out_id, out_name in self.output_mapping.items():
                o_name = postprocess_output_name(
                    out_name, self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False)
                if o_name not in self.outputs:
                    o_name = postprocess_output_name(
                        generate_layer_name(o_name, 'decoder_', with_prefix),
                        self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False)
                self.output_mapping[out_id] = o_name

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
    def predict(self, identifiers, input_data):
        feed_dict = self.prepare_inputs(input_data)
        return self.infer(feed_dict)

    def update_inputs_outputs_info(self, with_prefix):
        for input_id, input_name in self.input_mapping.items():
            self.input_mapping[input_id] = generate_layer_name(input_name, 'postnet_', with_prefix)
        if hasattr(self, 'outputs'):
            for out_id, out_name in self.output_mapping.items():
                o_name = postprocess_output_name(
                    out_name, self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False)
                if o_name not in self.outputs:
                    o_name = postprocess_output_name(
                    generate_layer_name(out_name, 'postnet_', with_prefix),
                    self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False)
                self.output_mapping[out_id] = o_name


class EncoderDLSDKModel(EncoderModel, TTSDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.input_mapping = {
            'text_encoder_outputs': 'text_encoder_outputs',
            'domain': 'domain',
            'f0s': 'f0s',
            'bert_embedding': 'bert_embedding'
        }
        self.output_mapping = {'encoder_outputs': 'encoder_outputs'}
        self.text_enc_dim = 384
        self.bert_dim = 768

    def prepare_inputs(self, feed):
        feed_dict = super().prepare_inputs(feed)
        if (
                self.input_mapping['text_encoder_outputs'] in self.dynamic_inputs or
                feed_dict[self.input_mapping['text_encoder_outputs']].shape !=
                self.inputs[self.input_mapping['text_encoder_outputs']].input_data.shape
        ):
            if not self.is_dynamic:
                new_shapes = {}
                for input_name in self.inputs:
                    new_shapes[input_name] = (
                        feed_dict[input_name].shape if input_name in feed_dict else self.inputs[input_name].shape)
                self._reshape_input(new_shapes)
        return feed_dict

    def infer(self, feed_dict):
        return self.exec_network.infer(feed_dict)


class EncoderOpenVINOModel(EncoderModel, TTSOVModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.input_mapping = {
            'text_encoder_outputs': 'text_encoder_outputs',
            'domain': 'domain',
            'f0s': 'f0s',
            'bert_embedding': 'bert_embedding'
        }
        self.output_mapping = {'encoder_outputs': 'encoder_outputs/sink_port_0'}
        self.text_enc_dim = 384
        self.bert_dim = 768

    def prepare_inputs(self, feed):
        feed_dict = super().prepare_inputs(feed)
        if (
                self.input_mapping['text_encoder_outputs'] in self.dynamic_inputs or
                feed_dict[self.input_mapping['text_encoder_outputs']].shape !=
                parse_partial_shape(self.inputs[self.input_mapping['text_encoder_outputs']].shape)
        ):
            if not self.is_dynamic:
                new_shapes = {}
                for input_name in self.inputs:
                    new_shapes[input_name] = (
                        feed_dict[input_name].shape if input_name in feed_dict else parse_partial_shape(
                            self.inputs[input_name].shape))
                self._reshape_input(new_shapes)
        return feed_dict


class EncoderONNXModel(BaseONNXModel, EncoderModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.input_mapping = {
            'text_encoder_outputs': 'text_encoder_outputs',
            'domain': 'domain',
            'f0s': 'f0s',
            'bert_embedding': 'bert_embedding'
        }
        self.output_mapping = {'encoder_outputs': 'encoder_outputs'}
        self.text_enc_dim = 384
        self.bert_dim = 768
        outputs = self.inference_session.get_outputs()
        self.output_names = [output.name for output in outputs]

    @property
    def inputs(self):
        inputs_info = self.inference_session.get_inputs()
        return {input_layer.name: input_layer.shape for input_layer in inputs_info}

    def infer(self, feed_dict):
        outs = self.inference_session.run(self.output_names, feed_dict)
        return dict(zip(self.output_names, outs))


class DecoderONNXModel(BaseONNXModel, DecoderModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
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
        outputs = self.inference_session.get_outputs()
        self.output_names = [output.name for output in outputs]

    @property
    def inputs(self):
        inputs_info = self.inference_session.get_inputs()
        return {input_layer.name: input_layer.shape for input_layer in inputs_info}

    def infer(self, feed_dict):
        outs = self.inference_session.run(self.output_names, feed_dict)
        return dict(zip(self.output_names, outs))

    @staticmethod
    def prepare_inputs(feed_dict):
        return feed_dict


class DecodeDLSDKModel(DecoderModel, TTSDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
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

    def infer(self, feed_dict):
        return self.exec_network.infer(feed_dict)

    def prepare_inputs(self, feed_dict):
        if next(iter(self.input_mapping.values())) not in feed_dict:
            feed_dict_ = {self.input_mapping[input_name]: data for input_name, data in feed_dict.items()}
            feed_dict = feed_dict_

        if (
                self.input_mapping['encoder_outputs'] in self.dynamic_inputs or
                feed_dict[self.input_mapping['encoder_outputs']].shape !=
                self.inputs[self.input_mapping['encoder_outputs']].input_data.shape
        ):
            if not self.is_dynamic:
                new_shapes = {}
                for input_name in self.inputs:
                    new_shapes[input_name] = (
                        feed_dict[input_name].shape if input_name in feed_dict else
                        self.inputs[input_name].input_data.shape)
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


class DecodeOpenVINOModel(DecoderModel, TTSOVModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
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
            'finished': '109/sink_port_0',
            'decoder_input': '108/sink_port_0',
            'attention_hidden': '68/sink_port_0',
            'attention_cell': '66/sink_port_0',
            'decoder_hidden': '106/sink_port_0',
            'decoder_cell': '104/sink_port_0',
            'attention_weights': '85/sink_port_0',
            'attention_weights_cum': '89/sink_port_0',
            'attention_context': '88/sink_port_0'
        }
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

    def prepare_inputs(self, feed_dict):
        if next(iter(self.input_mapping.values())) not in feed_dict:
            feed_dict_ = {self.input_mapping[input_name]: data for input_name, data in feed_dict.items()}
            feed_dict = feed_dict_

        if (
                self.input_mapping['encoder_outputs'] in self.dynamic_inputs or
                feed_dict[self.input_mapping['encoder_outputs']].shape !=
                parse_partial_shape(self.inputs[self.input_mapping['encoder_outputs']].get_partial_shape())
        ):
            if not self.is_dynamic:
                new_shapes = {}
                for input_name in self.inputs:
                    new_shapes[input_name] = (
                        feed_dict[input_name].shape if input_name in feed_dict else
                        parse_partial_shape(self.inputs[input_name].get_partial_shape()))
                self._reshape_input(new_shapes)

        if len(feed_dict) != len(self.inputs):
            extra_inputs = set(self.inputs).difference(set(feed_dict))
            for input_layer in extra_inputs:
                shape = parse_partial_shape(self.inputs[input_layer].get_partial_shape())
                if self.additional_inputs_filling == 'zeros':
                    feed_dict[input_layer] = np.zeros(shape, dtype=np.float32)
                else:
                    feed_dict[input_layer] = np.random.uniform(size=shape)

        return feed_dict


class PostNetONNXModel(BaseONNXModel, PostNetModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.input_mapping = {'mel_outputs': 'mel_outputs'}
        self.output_mapping = {'postnet_outputs': 'postnet_outputs'}
        outputs = self.inference_session.get_outputs()
        self.output_names = [output.name for output in outputs]

    @staticmethod
    def prepare_inputs(feed_dict):
        return feed_dict

    @property
    def inputs(self):
        inputs_info = self.inference_session.get_inputs()
        return {input_layer.name: input_layer.shape for input_layer in inputs_info}

    def infer(self, feed_dict):
        outs = self.inference_session.run(self.output_names, feed_dict)
        return dict(zip(self.output_names, outs))


class PostNetDLSDKModel(PostNetModel, TTSDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.input_mapping = {'mel_outputs': 'mel_outputs'}
        self.output_mapping = {'postnet_outputs': 'postnet_outputs'}

    def infer(self, feed_dict):
        return self.exec_network.infer(feed_dict)

    def prepare_inputs(self, feed_dict):
        input_shape = next(iter(feed_dict.values())).shape
        if input_shape != tuple(self.inputs[self.input_mapping['mel_outputs']].input_data.shape):
            self._reshape_input({self.input_mapping['mel_outputs']: input_shape})
        if next(iter(self.input_mapping.values())) not in feed_dict:
            return {self.input_mapping[input_name]: data for input_name, data in feed_dict.items()}
        return feed_dict


class PostNetOpenVINOModel(PostNetModel, TTSOVModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.input_mapping = {'mel_outputs': 'mel_outputs'}
        self.output_mapping = {'postnet_outputs': 'postnet_outputs/sink_port_0'}

    def prepare_inputs(self, feed_dict):
        input_shape = next(iter(feed_dict.values())).shape
        if input_shape != parse_partial_shape(self.inputs[self.input_mapping['mel_outputs']].get_partial_shape()):
            self._reshape_input({self.input_mapping['mel_outputs']: input_shape})
        if next(iter(self.input_mapping.values())) not in feed_dict:
            return {self.input_mapping[input_name]: data for input_name, data in feed_dict.items()}
        return feed_dict


class Tacotron2Evaluator(TextToSpeechEvaluator):
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        adapter_info = config['adapter']
        model = Synthesizer(
            config.get('network_info', {}), launcher, config.get('_models', []), adapter_info,
            config.get('_model_is_blob'), delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)
