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

from functools import partial
from collections import OrderedDict
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseCascadeModel, BaseDLSDKModel, create_model, BaseONNXModel, BaseOpenVINOModel
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, parse_partial_shape, postprocess_output_name


class OpenNMTEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = OpenNMTModel(
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
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


class OpenNMTModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['encoder', 'decoder', 'generator']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder, decoder and generator fields')

        self._encoder_mapping = {
                'dlsdk': EncoderDLSDKModel,
                'onnx_runtime': EncoderONNXModel,
                'openvino': EncoderOVModel
            }
        self._decoder_mapping = {
                'dlsdk': DecoderDLSDKModel,
                'onnx_runtime': DecoderONNXModel,
                'openvino': DecoderOVModel
            }

        self._generator_mapping = {
                'dlsdk': GeneratorDLSDKModel,
                'onnx_runtime': GeneratorONNXModel,
                'openvino': GeneratorOVModel
            }

        self.encoder = create_model(network_info['encoder'], launcher, self._encoder_mapping, 'encoder',
                                    delayed_model_loading)
        self.decoder = create_model(network_info['decoder'], launcher, self._decoder_mapping, 'decoder',
                                    delayed_model_loading)
        self.generator = create_model(network_info['generator'], launcher, self._generator_mapping, 'generator',
                                      delayed_model_loading)

        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder, 'generator': self.generator}
        self._raw_outs = OrderedDict()

        self.adapter = create_adapter(launcher.config.get('adapter', 'nmt'))

    def predict(self, identifiers, input_data, encoder_callback=None):
        predictions, raw_outputs = [], []

        for data in input_data:
            decode_strategy = BeamSearch(self.decoder.network_info)

            src_len = np.array([len(data)])
            h, c, memory, raw_outputs = self.encoder.predict(identifiers, {'src': np.array([[[t]] for t in data])})
            if encoder_callback:
                encoder_callback(raw_outputs)

            self.decoder.init_state(h, c, memory, src_len)
            self.decoder.tile_state(decode_strategy.beam_size)

            for _ in range(decode_strategy.max_length):
                decoder_input = decode_strategy.current_predictions.view().reshape([1, decode_strategy.beam_size, 1])

                decoder_output, _, raw_outputs = self.decoder.predict(identifiers, {'input': decoder_input})
                if encoder_callback:
                    encoder_callback(raw_outputs)

                log_probs, raw_outputs = self.generator.predict(identifiers, {'input': decoder_output.squeeze(axis=0)})
                if encoder_callback:
                    encoder_callback(raw_outputs)

                decode_strategy.advance(log_probs)
                any_finished = decode_strategy.is_finished.any()
                if any_finished:
                    decode_strategy.update_finished()
                    if decode_strategy.done:
                        break
                select_indices = decode_strategy.select_indices.squeeze()
                if any_finished:
                    self.decoder.reorder_state(select_indices, ('memory', 'mem_len'))
                if decode_strategy.beam_size > 1 or any_finished:
                    self.decoder.reorder_state(select_indices, ('h', 'c', 'input_feed'))

            prediction = np.array(decode_strategy.predictions)[..., :-1]
            prediction = self.adapter.process({'pred': np.transpose(prediction, [2, 1, 0])}, identifiers, None)
            predictions.append(prediction)

        return raw_outputs, predictions


class StatefulModel:
    state_names = []
    state_inputs = []
    state_outputs = []
    state = {}

    def init_state(self, *args):
        self.state = dict(zip(self.state_names, args))

    @staticmethod
    def preprocess_state(name, value):
        return value

    def fit_to_input(self, input_data):
        names = input_data.keys()
        for state_name, model_name in zip(self.state_names, self.state_inputs):
            if state_name not in names:
                value = self.preprocess_state(state_name, self.state[state_name])
                input_data[model_name] = value
        return super().fit_to_input(input_data)

    def propagate_output(self, data):
        for state_name, model_name in zip(self.state_names, self.state_outputs):
            if model_name in data.keys():
                self.state[state_name] = data[model_name]
        super().propagate_output(data)


class CommonDLSDKModel(BaseDLSDKModel):
    default_model_suffix = 'encoder'
    input_layers = []
    output_layers = []
    return_layers = []

    def predict(self, identifiers, input_data, callback=None):
        input_data = self.fit_to_input(input_data)
        results = self.exec_network.infer(input_data)
        self.propagate_output(results)
        names = self.return_layers if len(self.return_layers) > 0 else self.output_layers
        return tuple(results[name] for name in names) + (results,)

    def fit_to_input(self, input_data):
        if isinstance(input_data, dict):
            fitted = {}
            has_info = hasattr(self.exec_network, 'input_info')
            if has_info:
                input_info = self.exec_network.input_info
            else:
                input_info = self.exec_network.inputs
            for input_blob in input_info.keys():
                fitted.update(self.fit_one_input(input_blob, input_data[input_blob]))
        else:
            fitted = self.fit_one_input(self.input_blob, input_data)
        return fitted

    def fit_one_input(self, input_blob, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info[input_blob].input_data
        else:
            input_info = self.exec_network.inputs[input_blob]
        if tuple(input_info.shape) != np.shape(input_data):
            self._reshape_input({input_blob: np.shape(input_data)})

        return {input_blob: np.array(input_data)}

    def propagate_output(self, data):
        pass


class CommonOVModel(BaseOpenVINOModel):
    default_model_suffix = 'encoder'
    input_layers = []
    output_layers = []
    return_layers = []

    def predict(self, identifiers, input_data, callback=None):
        input_data = self.fit_to_input(input_data)
        results, raw_results = self.infer(input_data, raw_results=True)
        self.propagate_output(results)
        names = self.return_layers if len(self.return_layers) > 0 else self.output_layers
        return tuple(results[name] for name in names) + (raw_results,)

    def fit_to_input(self, input_data):
        if isinstance(input_data, dict):
            fitted = {}
            for input_blob in self.inputs:
                data_key = input_blob
                if input_blob.startswith(self.default_model_suffix):
                    data_key = input_blob.replace(self.default_model_suffix + '_', '')
                fitted.update(self.fit_one_input(input_blob, input_data[data_key]))
        else:
            fitted = self.fit_one_input(self.input_blob, input_data)
        return fitted

    def fit_one_input(self, input_blob, input_data):
        if (
            input_blob in self.dynamic_inputs or
            parse_partial_shape(self.inputs[input_blob].get_partial_shape()) != np.shape(input_data)
        ):
            self._reshape_input({input_blob: np.shape(input_data)})

        return {input_blob: np.array(input_data)}

    def propagate_output(self, data):
        pass

    def set_input_and_output(self):
        super().set_input_and_output()
        for idx, out in enumerate(self.output_layers):
            self.output_layers[idx] = postprocess_output_name(out, self.outputs,
                                                              additional_mapping=self.additional_output_mapping,
                                                              raise_error=False)
        if hasattr(self, 'return_layers'):
            for idx, out in enumerate(self.return_layers):
                self.return_layers[idx] = postprocess_output_name(out, self.outputs,
                                                                  additional_mapping=self.additional_output_mapping,
                                                                  raise_error=False)


class BeamSearch:
    def __init__(self, config):
        self.batch_size = config.get('batch', 1)
        self.pad = config.get('pad', 1)
        self.bos = config.get('bos', 2)
        self.eos = config.get('eos', 3)
        self.unk = config.get('unk', 4)
        self.n_best = config.get('nbest', 1)
        self.max_length = config.get('max_length', 100)
        self.min_length = config.get('min_length', 0)
        self.beam_size = config.get('beams', 5)

        # result caching
        self.predictions = [[] for _ in range(self.batch_size)]
        self.scores = [[] for _ in range(self.batch_size)]
        # self.attention = [[] for _ in range(batch_size)]
        self.hypotheses = [[] for _ in range(self.batch_size)]

        self.alive_attn = None

        n_paths = self.batch_size * self.beam_size
        self.forbidden_tokens = [{} for _ in range(n_paths)]

        # beam parameters
        self.top_beam_finished = np.zeros([self.batch_size], dtype=np.uint8)
        self._batch_offset = np.arange(self.batch_size, dtype="long")
        self.select_indices = None
        self.done = False

        self.alive_seq = np.full([self.batch_size * self.beam_size, 1], self.bos, dtype="long")
        self.is_finished = np.zeros((self.batch_size, self.beam_size), dtype=np.uint8)
        self.best_scores = np.full([self.batch_size], -1e10, dtype=float)
        self._beam_offset = np.arange(0, self.batch_size * self.beam_size, step=self.beam_size, dtype="long")
        self.topk_log_probs = np.asarray([0.0] + [float("-inf")] * (self.beam_size - 1))
        self.topk_log_probs = self.topk_log_probs.repeat(self.batch_size).reshape(self.batch_size, self.beam_size)
        self.topk_scores = np.empty((self.batch_size, self.beam_size), dtype=float)
        self.topk_ids = np.empty((self.batch_size, self.beam_size), dtype="long")
        self._batch_index = np.empty([self.batch_size, self.beam_size], dtype="long")

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def batch_offset(self):
        return self._batch_offset

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        if len(self) == self.max_length + 1:
            self.is_finished.fill(1)

    def _pick(self, log_probs):
        vocab_size = log_probs.shape[-1]
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)
        topk_ids = np.argsort(curr_scores)[..., range(self.beam_size * vocab_size - 1,
                                                      self.beam_size * (vocab_size - 1) - 1, -1)]
        topk_scores = curr_scores[..., topk_ids.squeeze(axis=0)]
        return topk_scores, topk_ids

    def update_finished(self):
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs[self.is_finished] = -1e10
        self.top_beam_finished |= self.is_finished[:, 0]
        predictions = self.alive_seq.view().reshape(_B_old, self.beam_size, step)
        non_finished_batch = []
        for i in range(self.is_finished.shape[0]):  # Batch level
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero()[0]
            for j in finished_hyp:  # Beam level: finished beam j in batch i
                self.hypotheses[b].append((self.topk_scores[i, j], predictions[i, j, 1:]))
            finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)  # ``(batch, n_best,)``
            else:
                non_finished_batch.append(i)

        non_finished = np.asarray(non_finished_batch)
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        self.remove_finished_batches(_B_new, _B_old, non_finished, predictions, step)

    def remove_finished_batches(self, _B_new, _B_old, non_finished, predictions, step):
        self.top_beam_finished = np.take(self.top_beam_finished, non_finished, axis=0)
        self._batch_offset = np.take(self._batch_offset, non_finished, axis=0)
        self.topk_log_probs = np.take(self.topk_log_probs, non_finished, axis=0)
        self._batch_index = np.take(self._batch_index, non_finished, axis=0)
        self.select_indices = self._batch_index.view().reshape([_B_new * self.beam_size])
        self.alive_seq = np.take(predictions, non_finished, axis=0).view().reshape([-1, self.alive_seq.shape[-1]])
        self.topk_scores = np.take(self.topk_scores, non_finished, axis=0)
        self.topk_ids = np.take(self.topk_ids, non_finished, axis=0)

    def advance(self, log_probs):
        vocab_size = log_probs.shape[-1]
        _B = log_probs.shape[0] // self.beam_size
        self.ensure_min_length(log_probs)
        log_probs += self.topk_log_probs.view().reshape((_B * self.beam_size, 1))
        length_penalty = 1

        curr_scores = log_probs / length_penalty
        self.topk_scores, self.topk_ids = self._pick(curr_scores)
        self.topk_log_probs = self.topk_scores * length_penalty
        self._batch_index = self.topk_ids // vocab_size
        self._batch_index += np.expand_dims(self._beam_offset[:_B], axis=1)
        self.select_indices = self._batch_index.view().reshape((_B * self.beam_size, 1))
        self.topk_ids = np.fmod(self.topk_ids, vocab_size)  # resolve true word ids

        self.alive_seq = np.concatenate(
            [np.take(self.alive_seq, self.select_indices.squeeze(axis=-1), 0),
             self.topk_ids.view().reshape((_B * self.beam_size, 1))], axis=-1)

        self.is_finished = np.equal(self.topk_ids, self.eos)
        self.ensure_max_length()


class CommonOpenNMTDecoder(StatefulModel):
    hidden_size = 500
    state_names = ['h', 'c', 'memory', 'mem_len', 'input_feed']
    beam_dim = [1, 1, 1, 0, 1]

    def init_state(self, h, c, memory, mem_len):
        super().init_state(h, c, memory, mem_len, None)

        batch_size = self.state["h"].shape[1]
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = np.expand_dims(np.zeros(h_size, dtype=np.float32), 0)

    @staticmethod
    def tile(x, count, dim):
        dims = [1] * len(x.shape)
        dims[dim] = count
        return np.tile(x, dims)

    def tile_state(self, count):
        for idx, name in enumerate(self.state_names):
            self.state[name] = self.tile(self.state[name], count, self.beam_dim[idx])

    def reorder_state(self, indices, names):
        for idx, name in enumerate(self.state_names):
            if name in names:
                self.state[name] = np.take(self.state[name], indices, axis=self.beam_dim[idx])

    @staticmethod
    def preprocess_state(name, value):
        return value.squeeze(0) if name == "input_feed" else value


class EncoderDLSDKModel(CommonDLSDKModel):
    default_model_suffix = 'encoder'
    input_layers = ['src', 'src_len']
    output_layers = ['state.0', 'state.1', 'memory']
    return_layers = ['state.0', 'state.1', 'memory']


class DecoderDLSDKModel(CommonOpenNMTDecoder, CommonDLSDKModel):
    default_model_suffix = 'decoder'
    input_layers = ['c_0', 'h_0', 'input', 'input_feed.1', 'mem_len', 'memory']
    output_layers = ['attn', 'c_1', 'h_1', 'input_feed', 'output']
    return_layers = ['output', 'attn']
    state_inputs = ['h_0', 'c_0', 'memory', 'mem_len', 'input_feed.1']
    state_outputs = ['h_1', 'c_1', '', '', 'input_feed']

    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        if network_info.get('outputs'):
            self.output_layers = network_info['outputs']
        if network_info.get('return_outputs'):
            self.return_layers = network_info['return_outputs']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class GeneratorDLSDKModel(CommonDLSDKModel):
    default_model_suffix = 'generator'
    input_layers = ['input']
    output_layers = ['output']


class EncoderOVModel(CommonOVModel):
    default_model_suffix = 'encoder'
    input_layers = ['src', 'src_len']
    output_layers = ['state.0', 'state.1', 'memory']
    return_layers = ['state.0', 'state.1', 'memory']


class DecoderOVModel(CommonOpenNMTDecoder, CommonOVModel):
    default_model_suffix = 'decoder'
    input_layers = ['c_0', 'h_0', 'input', 'input_feed.1', 'mem_len', 'memory']
    output_layers = ['attn', 'c_1', 'h_1', 'input_feed', 'output']
    return_layers = ['output', 'attn']
    state_inputs = ['h_0', 'c_0', 'memory', 'mem_len', 'input_feed.1']
    state_outputs = ['h_1/sink_port_0', 'c_1/sink_port_0', '', '', 'input_feed/sink_port_0']

    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        if network_info.get('outputs'):
            self.output_layers = network_info['outputs']
        if network_info.get('return_outputs'):
            self.return_layers = network_info['return_outputs']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class GeneratorOVModel(CommonOVModel):
    default_model_suffix = 'generator'
    input_layers = ['input']
    output_layers = ['output']


class CommonONNXModel(BaseONNXModel):
    default_model_suffix = 'encoder'
    input_layers = []
    output_layers = []
    return_layers = []

    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.input_blobs = self.inference_session.get_inputs()
        self.output_blobs = self.inference_session.get_outputs()

    def predict(self, identifiers, input_data, callback=None):
        fitted = self.fit_to_input(input_data)
        names = tuple(blob.name for blob in self.output_blobs)
        results = dict(zip(names, self.inference_session.run(names, fitted)))
        self.propagate_output(results)
        names = self.return_layers if len(self.return_layers) > 0 else self.output_layers
        return tuple(results[name] for name in names) + (results,)

    def fit_to_input(self, input_data):
        return {blob.name: input_data[blob.name] for blob in self.input_blobs}

    def propagate_output(self, data):
        pass


class EncoderONNXModel(CommonONNXModel):
    default_model_suffix = 'encoder'
    input_layers = ['src']
    output_layers = ['state.0', 'state.1', 'memory']


class DecoderONNXModel(CommonOpenNMTDecoder, CommonONNXModel):
    default_model_suffix = 'decoder'
    input_layers = ['c_0', 'h_0', 'input', 'input_feed.1', 'mem_len', 'memory']
    output_layers = ['attn', 'c_1', 'h_1', 'input_feed', 'output']
    return_layers = ['output', 'attn']
    state_inputs = ['h_0', 'c_0', 'memory', 'mem_len', 'input_feed.1']
    state_outputs = ['h_1', 'c_1', '', '', 'input_feed']


class GeneratorONNXModel(CommonONNXModel):
    default_model_suffix = 'generator'
    input_layers = ['input']
    output_layers = ['output']
