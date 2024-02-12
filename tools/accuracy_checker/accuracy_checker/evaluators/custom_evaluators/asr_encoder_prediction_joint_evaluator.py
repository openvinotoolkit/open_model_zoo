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

from pathlib import Path
import pickle  # nosec B403  # disable import-pickle check
from collections import OrderedDict
import numpy as np

from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, read_pickle, parse_partial_shape, postprocess_output_name
from .asr_encoder_decoder_evaluator import AutomaticSpeechRecognitionEvaluator
from .base_models import (
    BaseCascadeModel, BaseDLSDKModel, BaseOpenVINOModel, BaseONNXModel, create_model, create_encoder
)


class ASREvaluator(AutomaticSpeechRecognitionEvaluator):
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        adapter_info = config.get('adapter', 'dumb_decoder')
        model = ASRModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            adapter_info, delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)


class ASRModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, adapter_info, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['encoder', 'prediction', 'joint']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder, prediction and joint fields')
        self._encoder_mapping = {
            'dlsdk': EncoderDLSDKModel,
            'openvino': EncoderOVMOdel,
            'onnx_runtime': EncoderONNXModel,
            'dummy': DummyEncoder
        }
        self._prediction_mapping = {
            'dlsdk': PredictionDLSDKModel,
            'openvino': PredictionOVModel,
            'onnx_runtime': PredictionONNXModel
        }
        self._joint_mapping = {
            'dlsdk': JointDLSDKModel,
            'openvino': JointOVModel,
            'onnx_runtime': JointONNXModel
        }
        self.encoder = create_encoder(network_info['encoder'], launcher, self._encoder_mapping, delayed_model_loading)
        self.prediction = create_model(network_info['prediction'], launcher, self._prediction_mapping, 'prediction',
                                       delayed_model_loading)
        self.joint = create_model(network_info['joint'], launcher, self._joint_mapping, 'joint', delayed_model_loading)
        self.store_encoder_predictions = network_info['encoder'].get('store_predictions', False)
        self._encoder_predictions = [] if self.store_encoder_predictions else None
        self._part_by_name = {'encoder': self.encoder, 'prediction': self.prediction, 'joint': self.joint}
        self._raw_outs = OrderedDict()
        self.adapter = create_adapter(adapter_info)
        self._blank_id = 28
        self._sos = -1
        self._max_symbols_per_step = 30

    def predict(self, identifiers, input_data, encoder_callback=None):
        predictions, raw_outputs = [], []
        for data in input_data:
            encoder_prediction, decoder_inputs = self.encoder.predict(identifiers, data)
            if isinstance(encoder_prediction, tuple):
                encoder_prediction, raw_encoder_prediction = encoder_prediction
            else:
                raw_encoder_prediction = encoder_prediction
            if encoder_callback:
                encoder_callback(raw_encoder_prediction)
            if self.store_encoder_predictions:
                self._encoder_predictions.append(encoder_prediction)
            raw_output, prediction = self.decoder(identifiers, decoder_inputs, callback=encoder_callback)
            raw_outputs.append(raw_output)
            predictions.append(prediction)
        return raw_outputs, predictions

    def reset(self):
        self.processing_frames_buffer = []
        if self._encoder_predictions is not None:
            self._encoder_predictions = []

    def save_encoder_predictions(self):
        if self._encoder_predictions is not None:
            prediction_file = Path(self.network_info['encoder'].get('predictions', 'encoder_predictions.pickle'))
            with prediction_file.open('wb') as file:
                pickle.dump(self._encoder_predictions, file)

    def decoder(self, identifiers, logits, callback=None):
        output = []
        raw_outputs = []
        batches = logits.shape[0]
        for batch_idx in range(batches):
            inseq = np.squeeze(logits[batch_idx, :, :])
            # inseq: TxBxF
            logitlen = inseq.shape[0]
            sentence = self._greedy_decode(inseq, logitlen, callback)
            output.append(sentence)
        result = self.adapter.process(output, identifiers, [{}])

        return raw_outputs, result

    def _greedy_decode(self, x, out_len, callback=None):
        hidden_size = 320
        hidden = (np.zeros([2, 1, hidden_size]), np.zeros([2, 1, hidden_size]))
        label = []
        for time_idx in range(out_len):
            f = np.expand_dims(np.expand_dims(x[time_idx, ...], 0), 0)

            not_blank = True
            symbols_added = 0

            while not_blank and symbols_added < self._max_symbols_per_step:
                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden
                )
                if isinstance(g, tuple):
                    g, raw_g = g
                else:
                    raw_g = g
                if callback:
                    callback(raw_g)
                hidden_prime = (g[self.prediction.output_layers[0]], g[self.prediction.output_layers[1]])
                g = g[self.prediction.output_layers[2]]
                logp = self._joint_step(f, g, log_normalize=False, callback=callback)[0, :]

                k = np.argmax(logp)

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        return label

    def _pred_step(self, label, hidden):
        if label == self._sos:
            label = self._blank_id
        if label > self._blank_id:
            label -= 1
        inputs = {
            self.prediction.input_layers[0]: [[label, ]],
            self.prediction.input_layers[1]: hidden[0],
            self.prediction.input_layers[2]: hidden[1]
        }
        return self.prediction.predict(None, inputs)

    def _joint_step(self, enc, pred, log_normalize=False, callback=None):
        inputs = {self.joint.input_layers[0]: enc, self.joint.input_layers[1]: pred}
        logits, logits_blob = self.joint.predict(None, inputs)
        if isinstance(logits, tuple):
            logits, raw_logits = logits
        else:
            raw_logits = logits
        if callback:
            callback(raw_logits)
        logits = logits_blob[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = np.argmax(np.log(logits), axis=len(logits.shape) - 1)
        return probs

    def _get_last_symb(self, labels) -> int:
        return self._sos if len(labels) == 0 else labels[-1]


class CommonDLSDKModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.input_layers = network_info.get('inputs', self.default_input_layers)
        self.output_layers = network_info.get('outputs', self.default_output_layers)
        if len(self.input_layers) == 1:
            self.input_blob = self.input_layers[0]
        if len(self.output_layers) == 1:
            self.output_blob = self.output_layers[0]
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data, callback=None):
        input_data = self.fit_to_input(input_data)
        results = self.exec_network.infer(input_data)
        return results, results[self.output_blob]

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
        if input_blob in self.dynamic_inputs or tuple(input_info.shape) != np.shape(input_data):
            self._reshape_input({input_blob: np.shape(input_data)})

        return {input_blob: np.array(input_data)}

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.output_blob is None:
                output_blob = next(iter(self.exec_network.outputs))
            else:
                output_blob = (
                    '_'.join([self.default_model_suffix, self.output_blob])
                    if with_prefix else self.output_blob.split(self.default_model_suffix + '_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix
            for idx, inp in enumerate(self.input_layers):
                self.input_layers[idx] = (
                    '_'.join([self.default_model_suffix, inp])
                    if with_prefix else inp.split(self.default_model_suffix)[-1]
                )
            for idx, out in enumerate(self.output_layers):
                self.output_layers[idx] = (
                    '_'.join([self.default_model_suffix, out])
                    if with_prefix else out.split(self.default_model_suffix)[-1]
                )


class CommonOVModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.input_layers = network_info.get('inputs', self.default_input_layers)
        self.output_layers = network_info.get('outputs', self.default_output_layers)
        if len(self.input_layers) == 1:
            self.input_blob = self.input_layers[0]
        if len(self.output_layers) == 1:
            self.output_blob = self.output_layers[0]
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data, callback=None):
        input_data = self.fit_to_input(input_data)
        results = self.infer(input_data, raw_results=True)
        return results, results[self.output_blob] if not isinstance(results, tuple) else results[0][self.output_blob]

    def fit_to_input(self, input_data):
        if isinstance(input_data, dict):
            fitted = {}
            for input_blob in self.inputs.keys():
                fitted.update(self.fit_one_input(input_blob, input_data[input_blob]))
        else:
            fitted = self.fit_one_input(self.input_blob, input_data)
        return fitted

    def fit_one_input(self, input_blob, input_data):
        if (input_blob in self.dynamic_inputs or parse_partial_shape(
            self.inputs[input_blob].get_partial_shape()) != np.shape(input_data)):
            self._reshape_input({input_blob: np.shape(input_data)})

        return {input_blob: np.array(input_data)}

    def set_input_and_output(self):
        input_blob = next(iter(self.inputs))
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.output_blob is None:
                output_blob = next(iter(self.outputs))
            else:
                output_blob = postprocess_output_name(self.output_blob, self.outputs, raise_error=False)

            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix
            for idx, inp in enumerate(self.input_layers):
                self.input_layers[idx] = (
                    '_'.join([self.default_model_suffix, inp])
                    if with_prefix else inp.split(self.default_model_suffix)[-1]
                )
        for idx, out in enumerate(self.output_layers):
            self.output_layers[idx] = postprocess_output_name(out, self.outputs, raise_error=False)


class EncoderDLSDKModel(CommonDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_input_layers = []
        self.default_output_layers = ['472']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class EncoderOVMOdel(CommonOVModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_input_layers = []
        self.default_output_layers = ['472/sink_port_0']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class PredictionDLSDKModel(CommonDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_input_layers = ['input.1', '1', '2']
        self.default_output_layers = ['151', '152', '153']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class PredictionOVModel(CommonOVModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_input_layers = ['input.1', '1', '2']
        self.default_output_layers = ['151/sink_port_0', '152/sink_port_0', '153/sink_port_0']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class JointDLSDKModel(CommonDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_input_layers = ['0', '1']
        self.default_output_layers = []
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class JointOVModel(CommonOVModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_input_layers = ['0', '1']
        self.default_output_layers = []
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class CommonONNXModel(BaseONNXModel):
    def predict(self, identifiers, input_data, callback=None):
        fitted = self.fit_to_input(input_data)
        results = self.inference_session.run((self.output_blob.name, ), fitted)
        return results, results[0]


class EncoderONNXModel(CommonONNXModel):
    def fit_to_input(self, input_data):
        frames, _, _ = input_data.shape
        return {self.input_blob.name: input_data, '1': np.array([frames], dtype=int)}


class PredictionONNXModel(CommonONNXModel):
    pass


class JointONNXModel(CommonONNXModel):
    pass


class DummyEncoder:
    def __init__(self, network_info, launcher):
        self.network_info = network_info
        self.launcher = launcher
        if 'predictions' not in network_info:
            raise ConfigError('predictions_file is not found')
        self._predictions = read_pickle(network_info['predictions'])
        self.iterator = 0

    def predict(self, identifiers, input_data):
        result = self._predictions[self.iterator]
        self.iterator += 1
        return None, result

    def release(self):
        pass
