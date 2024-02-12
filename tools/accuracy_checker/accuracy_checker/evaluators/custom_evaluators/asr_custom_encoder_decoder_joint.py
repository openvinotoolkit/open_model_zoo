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
import heapq
import math
import numpy as np
from .asr_encoder_prediction_joint_evaluator import ASREvaluator
from .base_models import create_model, BaseCascadeModel, BaseDLSDKModel, BaseONNXModel, BaseOpenVINOModel
from ...adapters import create_adapter
from ...utils import generate_layer_name, contains_all
from ...config import ConfigError


class BeamEntry:
    def __init__(self, blank, other=None, hidden=None):
        if other is None:
            self.sequence = [blank]
            self.log_prob = 0.0
            self.cache = []
            self.hidden = hidden
        else:
            self.sequence = other.sequence[:]
            self.log_prob = other.log_prob
            self.cache = other.cache[:]
            self.hidden = other.hidden

    def __lt__(self, other):
        return self.log_prob < other.log_prob

    def __eq__(self, other):
        return self.log_prob == other.log_prob

    def is_prefix(self, other):
        if self.sequence == other.sequence or len(self.sequence) >= len(other.sequence):
            return False
        for i, s in enumerate(self.sequence):
            if s != other.sequence[i]:
                return False
        return True


class Encoder:
    def reset(self):
        self.h0 = np.zeros((6, 1, 1024)).astype('float32')
        self.c0 = np.zeros((6, 1, 1024)).astype('float32')

    def predict(self, identifiers, input_data):
        # Evaluate the encoder network one feature frame at a time
        data = self.fit_to_input(input_data)
        outputs = self.infer(data)
        if isinstance(outputs, tuple):
            outputs, raw_outputs = outputs
        else:
            raw_outputs = outputs
        encoder_output = outputs[self.encoder_out]
        self.h0 = outputs[self.h0_out]
        self.c0 = outputs[self.c0_out]
        return encoder_output.squeeze(), raw_outputs

    def fit_to_input(self, input_data):
        return {self.input: input_data, self.h0_input: self.h0, self.c0_input: self.c0}

    @property
    def input_names(self):
        return [self.input, self.h0_input, self.c0_input]

    @input_names.setter
    def input_names(self, list_inputs):
        assert len(list_inputs) == 3
        self.input, self.h0_input, self.c0_input = list_inputs

    @property
    def output_names(self):
        return [self.encoder_out, self.h0_out, self.c0_out]

    @output_names.setter
    def output_names(self, list_outputs):
        assert len(list_outputs) == 3
        self.encoder_out, self.h0_out, self.c0_out = list_outputs

    def select_inputs_outputs(self, network_info):
        input_info = network_info.get('inputs', {})
        if isinstance(input_info, list):
            self.input_names = input_info
        else:
            input_list = [
                input_info.get('input', self.default_inputs[0]),
                input_info.get('h0_input', self.default_inputs[1]),
                input_info.get('c0_input', self.default_inputs[2])
            ]
            self.input_names = input_list
        output_info = network_info.get('outputs', {})
        if isinstance(output_info, list):
            self.output_names = output_info
        else:
            output_list = [
                input_info.get('output', self.default_outputs[0]),
                input_info.get('h0_output', self.default_outputs[1]),
                input_info.get('c0_output', self.default_outputs[2])
            ]
            self.output_names = output_list


class Decoder:
    def reset(self):
        self.h0 = np.zeros((2, 1, 1024)).astype('float32')
        self.c0 = np.zeros((2, 1, 1024)).astype('float32')

    def predict(self, identifiers, input_data, hidden=None):
        data = self.fit_to_input(input_data, hidden)
        outputs = self.infer(data)
        if isinstance(outputs, tuple):
            outputs, raw_outputs = outputs
        else:
            raw_outputs = outputs
        self.h0 = outputs[self.h0_out]
        self.c0 = outputs[self.c0_out]
        return outputs[self.decoder_out].squeeze(), (self.h0, self.c0), raw_outputs

    def fit_to_input(self, token_id, hidden):
        if hidden is None:
            self.reset()
        else:
            self.h0 = hidden[0]
            self.c0 = hidden[1]
        input_data =  np.array([token_id]).astype('int64')
        return {self.input: input_data, self.h0_input: self.h0, self.c0_input: self.c0}

    @property
    def input_names(self):
        return [self.input, self.h0_input, self.c0_input]

    @input_names.setter
    def input_names(self, list_inputs):
        assert len(list_inputs) == 3
        self.input, self.h0_input, self.c0_input = list_inputs

    @property
    def output_names(self):
        return [self.decoder_out, self.h0_out, self.c0_out]

    @output_names.setter
    def output_names(self, list_outputs):
        assert len(list_outputs) == 3
        self.decoder_out, self.h0_out, self.c0_out = list_outputs

    def select_inputs_outputs(self, network_info):
        input_info = network_info.get('inputs', {})
        if isinstance(input_info, list):
            self.input_names = input_info
        else:
            input_list = [
                input_info.get('input', self.default_inputs[0]),
                input_info.get('h0_input', self.default_inputs[1]),
                input_info.get('c0_input', self.default_inputs[2])
            ]
            self.input_names = input_list
        output_info = network_info.get('outputs', {})
        if isinstance(output_info, list):
            self.output_names = output_info
        else:
            output_list = [
                input_info.get('output', self.default_outputs[0]),
                input_info.get('h0_output', self.default_outputs[1]),
                input_info.get('c0_output', self.default_outputs[2])
            ]
            self.output_names = output_list


class Joint:
    def predict(self, identifiers, input_data):
        encoder_out, predictor_out = input_data
        data = self.fit_to_input(encoder_out, predictor_out)
        outputs = self.infer(data)
        if isinstance(outputs, tuple):
            outputs, raw_outputs = outputs
        else:
            raw_outputs = outputs
        joint_out = outputs[self.output]
        return log_softmax(joint_out), raw_outputs

    def fit_to_input(self, encoder_out, predictor_out):
        return {self.input1: encoder_out, self.input2: predictor_out}

    @property
    def input_names(self):
        return [self.input1, self.input2]

    @input_names.setter
    def input_names(self, list_inputs):
        assert len(list_inputs) == 2
        self.input1, self.input2 = list_inputs

    @property
    def output_names(self):
        return [self.output]

    @output_names.setter
    def output_names(self, list_outputs):
        assert len(list_outputs) == 1
        self.output = list_outputs[0]

    def select_inputs_outputs(self, network_info):
        input_info = network_info.get('inputs', {})
        if isinstance(input_info, list):
            self.input_names = input_info
        else:
            input_list = [
                input_info.get('input1', self.default_inputs[0]),
                input_info.get('input2', self.default_inputs[1]),
            ]
            self.input_names = input_list
        output_info = network_info.get('outputs', {})
        if isinstance(output_info, list):
            self.output_names = output_info
        else:
            output_list = [
                input_info.get('output', self.default_outputs[0]),
            ]
            self.output_names = output_list

    def reset(self):
        pass


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


class CommonDLSDKModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.select_inputs_outputs(network_info)
        self.reset()
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def set_input_and_output(self):
        if self.exec_network is not None:
            has_info = hasattr(self.exec_network, 'input_info')
            input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        else:
            has_info = hasattr(self.network, 'input_info')
            input_info = self.network.input_info if has_info else self.network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if with_prefix != self.with_prefix:
            self.input_names = [
                generate_layer_name(
                    inp_name, self.default_model_suffix + '_', with_prefix) for inp_name in self.input_names
            ]
            self.output_names = [
                generate_layer_name(
                    out_name, self.default_model_suffix + '_', with_prefix) for out_name in self.output_names
            ]
            self.with_prefix = with_prefix

    def infer(self, input_data):
        return self.exec_network.infer(input_data)

    def predict(self, identifiers, input_data):
        raise NotImplementedError


class CommonOpenVINOModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.select_inputs_outputs(network_info)
        self.reset()
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def set_input_and_output(self):
        inputs = self.exec_network.inputs if self.exec_network is not None else self.network.inputs
        input_blob = next(iter(inputs)).get_node().friendly_name
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if with_prefix != self.with_prefix:
            self.input_names = [
                generate_layer_name(
                    inp_name, self.default_model_suffix + '_', with_prefix) for inp_name in self.input_names
            ]
            self.output_names = [
                generate_layer_name(
                    out_name, self.default_model_suffix + '_', with_prefix) for out_name in self.output_names
            ]
            self.with_prefix = with_prefix

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def infer(self, input_data, raw_results=False):
        return super().infer(input_data, True)


class DLSDKEncoder(Encoder, CommonDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['input_0', 'input_1', 'input_2']
        self.default_outputs = ['output_0', 'output_1', 'output_2']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class OVEncoder(Encoder, CommonOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['input_0', 'input_1', 'input_2']
        self.default_outputs = ['output_0/sink_port_0', 'output_1/sink_port_0', 'output_2/sink_port_0']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class DLSDKDecoder(Decoder, CommonDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['input_0', 'input_1', 'input_2']
        self.default_outputs = ['output_0', 'output_1', 'output_2']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class OVDecoder(Decoder, CommonOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['input_0', 'input_1', 'input_2']
        self.default_outputs = ['output_0/sink_port_0', 'output_1/sink_port_0', 'output_2/sink_port_0']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class DLSDKJoint(Joint, CommonDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['0', '1']
        self.default_outputs = ['8']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class OVJoint(Joint, CommonOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['0', '1']
        self.default_outputs = ['8/sink_port_0']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class CommonONNXModel(BaseONNXModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.select_inputs_outputs(network_info)
        self.reset()
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def infer(self, input_data):
        results = self.inference_session.run(self.output_names, input_data)
        return dict(zip(self.output_names, results))


class ONNXEncoder(Encoder, CommonONNXModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['input_0', 'input_1', 'input_2']
        self.default_outputs = ['output_0', 'output_1', 'output_2']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class ONNXDecoder(Decoder, CommonONNXModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['input_0', 'input_1', 'input_2']
        self.default_outputs = ['output_0', 'output_1', 'output_2']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class ONNXJoint(Joint, CommonONNXModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.default_inputs = ['0', '1']
        self.default_outputs = ['8']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class CustomASREvaluator(ASREvaluator):
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, launcher_config = cls.get_dataset_and_launcher_info(config)
        adapter_config = launcher_config.get('adapter', {'type': 'dumb_decoder'})
        model = ASRModel(
            config.get('network_info', {}), adapter_config, launcher, config.get('_models', []),
            config.get('_model_is_blob'), delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)


class ASRModel(BaseCascadeModel):
    beam_width = 5

    def __init__(self, network_info, adapter_config, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['encoder', 'decoder', 'joint']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder, decoder and joint fields')
        self._decoder_mapping = {
            'dlsdk': DLSDKDecoder,
            'openvino': OVDecoder,
            'onnx_runtime': ONNXDecoder
        }
        self._encoder_mapping = {
            'dlsdk': DLSDKEncoder,
            'openvino': OVEncoder,
            'onnx_runtime': ONNXEncoder
        }
        self._joint_mapping = {
            'dlsdk': DLSDKJoint,
            'openvino': OVJoint,
            'onnx_runtime': ONNXJoint
        }
        self.encoder = create_model(network_info['encoder'], launcher, self._encoder_mapping, 'encoder',
                                    delayed_model_loading)
        self.decoder = create_model(network_info['decoder'], launcher, self._decoder_mapping, 'decoder',
                                    delayed_model_loading)
        self.joint = create_model(network_info['joint'], launcher, self._joint_mapping, 'joint', delayed_model_loading)
        self.adapter = create_adapter(adapter_config)
        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder, 'joint': self.joint}

    def predict(self, identifiers, input_data, encoder_callback=None):
        input_data = self.prepare_records(input_data)
        B = [BeamEntry(blank=self.adapter.blank)]
        self.encoder.reset()
        self.decoder.reset()
        for idx in range(0, input_data.shape[1], 3):
            encoder_output, raw_outputs = self.encoder.predict(identifiers, input_data[:, idx])
            if encoder_callback is not None:
                encoder_callback(raw_outputs)
            A = B
            B = []
            for j in range(len(A) - 1):
                for i in range(j + 1, len(A)):
                    if A[i].is_prefix(A[j]):
                        A = self.fill_prefix(A, i, j, encoder_output, encoder_callback)
            while True:
                y_hat = max(A)
                A.remove(y_hat)
                decoder_output, hidden, raw_outputs = self.decoder.predict(identifiers, y_hat.sequence[-1],
                                                                           hidden=y_hat.hidden)
                if encoder_callback is not None:
                    encoder_callback(raw_outputs)
                joint_output, raw_outputs = self.joint.predict(identifiers, (encoder_output, decoder_output))
                if encoder_callback is not None:
                    encoder_callback(raw_outputs)
                joint_output = self.handle_eos(joint_output)
                A = self.fill_beam(A, B, y_hat, hidden, joint_output, decoder_output)
                y_hat = max(A)
                yb = max(B)
                if len(B) >= self.beam_width and yb.log_prob >= y_hat.log_prob:
                    break
            B = heapq.nlargest(self.beam_width, B)
        return [{}], self.adapter.process([B[0].sequence], identifiers, [{}])

    @staticmethod
    def prepare_records(features):
        feats = features[0].reshape(features[0].shape[0] // 80, 80).T
        feat_stack = np.vstack(
            (feats[:, 0:-7],
             feats[:, 1:-6],
             feats[:, 2:-5],
             feats[:, 3:-4],
             feats[:, 4:-3],
             feats[:, 5:-2],
             feats[:, 6:-1],
             feats[:, 7:]))

        return feat_stack

    def fill_beam(self, A, B, y_hat, hidden, joint_output, decoder_output):
        for k, _ in enumerate(self.adapter.alphabet):
            yk = BeamEntry(self.adapter.blank, y_hat)
            yk.log_prob += joint_output[k]
            if k == self.adapter.blank:
                heapq.heappush(B, yk)
                continue
            yk.hidden = hidden
            yk.sequence.append(k)
            yk.cache.append(decoder_output)
            A.append(yk)
        return A

    def handle_eos(self, joint_output):
        if self.adapter.eos != -1:
            joint_output[self.adapter.blank] = np.log(
                np.exp(joint_output[self.adapter.blank]) + np.exp(joint_output[self.adapter.eos]))
            joint_output[self.adapter.eos] = np.log(1e-10)
        return joint_output

    def fill_prefix(self, A, i, j, encoder_output, callback=None):
        def log_add(a, b):
            return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))
        decoder_output, _, raw_outputs = self.decoder.predict(None, A[i].sequence[-1], hidden=A[i].hidden)
        if callback is not None:
            callback(raw_outputs)
        idx = len(A[i].sequence)
        joint_output, raw_outputs = self.joint.predict(None, (encoder_output, decoder_output))
        if callback is not None:
            callback(raw_outputs)
        joint_output = self.handle_eos(joint_output)
        curlogp = A[i].log_prob + float(joint_output[A[j].sequence[idx]])
        for k in range(idx, len(A[j].sequence) - 1):
            joint_output, raw_outputs = self.joint.predict(None, (encoder_output, A[j].cache[k]))
            if callback is not None:
                callback(raw_outputs)
            curlogp += joint_output[A[j].sequence[k + 1]]
        A[j].log_prob = log_add(A[j].log_prob, curlogp)
        return A
