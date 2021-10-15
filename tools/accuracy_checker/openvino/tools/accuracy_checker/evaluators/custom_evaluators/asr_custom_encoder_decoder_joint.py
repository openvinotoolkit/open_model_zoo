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
import heapq
import math
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .asr_encoder_prediction_joint_evaluator import ASREvaluator
from ...adapters import create_adapter
from ...utils import generate_layer_name, contains_all, contains_any, get_path
from ...logging import print_info
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


class BaseModel:
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher
        self.select_inputs_outputs(network_info)
        self.reset()
        if not delayed_model_loading:
            self.prepare_model(network_info, launcher)

    def infer(self, input_data):
        raise NotImplementedError

    def release(self):
        pass

    def select_inputs_outputs(self, network_info):
        raise NotImplementedError

    def reset(self):
        pass

class Encoder(BaseModel):
    default_model_suffix = 'encoder'
    default_inputs = ['input_0', 'input_1', 'input_2']
    default_outputs = ['output_0', 'output_1', 'output_2']

    def reset(self):
        self.h0 = np.zeros((6, 1, 1024)).astype('float32')
        self.c0 = np.zeros((6, 1, 1024)).astype('float32')

    def predict(self, features):
        # Evaluate the encoder network one feature frame at a time
        input_data = self.fit_to_input(features)
        outputs = self.infer(input_data)
        encoder_output = np.array(outputs[self.encoder_out]).squeeze()
        self.h0 = outputs[self.h0_out]
        self.c0 = outputs[self.c0_out]
        return encoder_output, outputs

    def infer(self, input_data):
        raise NotImplementedError

    def fit_to_input(self, features):
        return {self.input: features, self.h0_input: self.h0, self.c0_input: self.c0}

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


class Decoder(BaseModel):
    default_model_suffix = 'decoder'
    default_inputs = ['input_0', 'input_1', 'input_2']
    default_outputs = ['output_0', 'output_1', 'output_2']

    def reset(self):
        self.h0 = np.zeros((2, 1, 1024)).astype('float32')
        self.c0 = np.zeros((2, 1, 1024)).astype('float32')

    def predict(self, token_id, hidden=None):
        input_data = self.fit_to_input(token_id, hidden)
        outputs = self.infer(input_data)
        self.h0 = outputs[self.h0_out]
        self.c0 = outputs[self.c0_out]
        return np.array(outputs[self.decoder_out]).squeeze(), (self.h0, self.c0), outputs

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

    def infer(self, input_data):
        raise NotImplementedError

class Joint(BaseModel):
    default_model_suffix = 'joint'
    default_inputs = ['0', '1']
    default_outputs = ['8']

    def predict(self, encoder_out, predictor_out):
        input_data = self.fit_to_input(encoder_out, predictor_out)
        outputs = self.infer(input_data)
        joint_out = outputs[self.output]
        return log_softmax(np.array(joint_out).squeeze()), outputs

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

    def infer(self, input_data):
        raise NotImplementedError

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


class CommonONNXModel(BaseModel):
    def prepare_model(self, network_info, launcher):
        model = self.automatic_model_search(network_info)
        self.inference_session = launcher.create_inference_session(str(model))

    def infer(self, input_data):
        results = self.inference_session.run(self.output_names, input_data)
        return dict(zip(self.output_names, results))

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

    def select_inputs_outputs(self, network_info):
        pass

class ONNXEncoder(CommonONNXModel, Encoder):
    pass


class ONNXDecoder(CommonONNXModel, Decoder):
    pass


class ONNXJoint(CommonONNXModel, Joint):
    pass


class CommonDLSDKModel:
    with_prefix = None

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

    def prepare_model(self, network_info, launcher):
        self.load_model(network_info, launcher, True)

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
        accepted_suffixes = ['.blob', '.xml', '.onnx']
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

    def infer(self, input_data):
        return self.exec_network.infer(input_data)


class DLSDKEncoder(CommonDLSDKModel, Encoder):
    pass


class DLSDKDecoder(CommonDLSDKModel, Decoder):
    pass


class DLSDKJoint(CommonDLSDKModel, Joint):
    pass


def create_encoder(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': DLSDKEncoder,
        'onnx_runtime': ONNXEncoder
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


def create_decoder(model_config, launcher, delayed_model_loading):
    launcher_model_mapping = {
        'dlsdk': DLSDKDecoder,
        'onnx_runtime': ONNXDecoder
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


def create_joint(model_config, launcher, delayed_model_loading):
    launcher_model_mapping = {
        'dlsdk': DLSDKJoint,
        'onnx_runtime': ONNXJoint
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


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


class ASRModel:
    beam_width = 5

    def __init__(self, network_info, adapter_config, launcher, models_args, is_blob, delayed_model_loading=False):
        if models_args and not delayed_model_loading:
            encoder = network_info.get('encoder', {})
            decoder = network_info.get('decoder', {})
            joint = network_info.get('joint', {})
            if not contains_any(encoder, ['model', 'onnx_model']) and models_args:
                encoder['model'] = models_args[0]
                encoder['_model_is_blob'] = is_blob
            if not contains_any(decoder, ['model', 'onnx_model']) and models_args:
                decoder['model'] = models_args[1 if len(models_args) > 1 else 0]
                decoder['_model_is_blob'] = is_blob
            if not contains_any(joint, ['model', 'onnx_model']) and models_args:
                joint['model'] = models_args[2 if len(models_args) > 2 else 0]
                joint['_model_is_blob'] = is_blob
            network_info.update({'encoder': encoder, 'decoder': decoder, 'joint': joint})
        if not contains_all(network_info, ['encoder', 'decoder', 'joint']) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder, prediction and joint fields')
        self.encoder = create_encoder(network_info['encoder'], launcher, delayed_model_loading)
        self.decoder = create_decoder(network_info['decoder'], launcher, delayed_model_loading)
        self.joint = create_joint(network_info['joint'], launcher, delayed_model_loading)
        self.adapter = create_adapter(adapter_config)
        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder, 'joint': self.joint}

    def release(self):
        self.encoder.release()
        self.decoder.release()
        self.joint.release()

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, launcher)

    def get_network(self):
        return [{'name': 'encoder', 'model': self.encoder.network},
                {'name': 'decoder', 'model': self.decoder.network},
                {'name': 'joint', 'model': self.joint.network}]

    def predict(self, identifiers, input_data, encoder_callback=None):
        input_data = self.prepare_records(input_data)
        B = [BeamEntry(blank=self.adapter.blank)]
        self.encoder.reset()
        self.decoder.reset()
        for idx in range(0, input_data.shape[1], 3):
            encoder_output, raw_outputs = self.encoder.predict(input_data[:, idx])
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
                decoder_output, hidden, raw_outputs = self.decoder.predict(y_hat.sequence[-1], hidden=y_hat.hidden)
                if encoder_callback is not None:
                    encoder_callback(raw_outputs)
                joint_output, raw_outputs = self.joint.predict(encoder_output, decoder_output)
                if encoder_callback is not None:
                    encoder_callback(raw_outputs)
                joint_output = self.handle_eos(joint_output)
                A = self.fill_beam(A, B, y_hat, hidden, joint_output, decoder_output)
                y_hat = max(A)
                yb = max(B)
                if len(B) >= self.beam_width and yb.log_prob >= y_hat.log_prob:
                    break
            B = heapq.nlargest(self.beam_width, B)
            return self.adapter.process([B[0].sequence], identifiers, [{}]), {}

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
        decoder_output, _, raw_outputs = self.decoder.predict(A[i].sequence[-1], hidden=A[i].hidden)
        if callback is not None:
            callback(raw_outputs)
        idx = len(A[i].sequence)
        joint_output, raw_outputs = self.joint.predict(encoder_output, decoder_output)
        if callback is not None:
            callback(raw_outputs)
        joint_output = self.handle_eos(joint_output)
        curlogp = A[i].log_prob + float(joint_output[A[j].sequence[idx]])
        for k in range(idx, len(A[j].sequence) - 1):
            joint_output, raw_outputs = self.joint.predict(encoder_output, A[j].cache[k])
            if callback is not None:
                callback(raw_outputs)
            curlogp += joint_output[A[j].sequence[k + 1]]
        A[j].log_prob = log_add(A[j].log_prob, curlogp)
        return A
