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
from collections import OrderedDict
from functools import partial
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations
from ...logging import print_info
from ...representation import CharacterRecognitionPrediction, CharacterRecognitionAnnotation


class TextRecognitionWithAttentionEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, lowercase, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        self.lowercase = lowercase

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        lowercase = config.get('lowercase', False)
        model_type = config.get('model_type', 'SequentialFormulaRecognitionModel')
        if model_type not in MODEL_TYPES.keys():
            raise ValueError(f'Model type {model_type} is not supported')
        model = MODEL_TYPES[model_type](
            config.get('network_info', {}), launcher, config.get('_models', []), {}, config.get('_model_is_blob'),
            delayed_model_loading=delayed_model_loading
        )
        return cls(dataset_config, launcher, model, lowercase, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_data, batch_meta = extract_image_representations(batch_inputs)
            temporal_output_callback = None
            if output_callback:
                temporal_output_callback = partial(output_callback, metrics_result=None,
                                                   element_identifiers=batch_identifiers,
                                                   dataset_indices=batch_input_ids)
            batch_prediction, batch_raw_prediction = self.model.predict(
                batch_identifiers, batch_data, callback=temporal_output_callback
            )
            if self.lowercase:
                batch_prediction = batch_prediction.lower()
                batch_annotation = [CharacterRecognitionAnnotation(
                    label=ann.label.lower(), identifier=ann.identifier) for ann in batch_annotation]
            batch_prediction = [CharacterRecognitionPrediction(
                label=batch_prediction, identifier=batch_annotation[0].identifier)]
            batch_annotation, batch_prediction = self.postprocessor.process_batch(
                batch_annotation, batch_prediction, batch_meta
            )
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)

    def reset(self):
        super().reset()
        self.model.reset()

    def select_dataset(self, dataset_tag):
        super().select_dataset(dataset_tag)
        if self.model.vocab is None:
            self.model.vocab = self.dataset.metadata.get('vocab')


class BaseModel:
    def __init__(self, network_info, launcher, default_model_suffix):
        self.default_model_suffix = default_model_suffix
        self.network_info = network_info

    def predict(self, inputs, identifiers=None):
        raise NotImplementedError

    def release(self):
        pass

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
        weights = Path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(self.default_model_suffix, weights))

        return model, weights

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


def create_recognizer(
        model_config, launcher, suffix, delayed_model_loading=False, inputs_mapping=None, outputs_mapping=None
):
    launcher_model_mapping = {
        'dlsdk': RecognizerDLSDKModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, suffix,
                       delayed_model_loading=delayed_model_loading,
                       inputs_mapping=inputs_mapping, outputs_mapping=outputs_mapping)


class BaseSequentialModel:
    def __init__(self, network_info, launcher, models_args, meta, is_blob=None, delayed_model_loading=False):
        recognizer_encoder = network_info.get('recognizer_encoder', {})
        recognizer_decoder = network_info.get('recognizer_decoder', {})
        if not delayed_model_loading:
            if 'model' not in recognizer_encoder:
                recognizer_encoder['model'] = models_args[0]
                recognizer_encoder['_model_is_blob'] = is_blob
            if 'model' not in recognizer_decoder:
                recognizer_decoder['model'] = models_args[len(models_args) == 2]
                recognizer_decoder['_model_is_blob'] = is_blob
            network_info.update({
                'recognizer_encoder': recognizer_encoder,
                'recognizer_decoder': recognizer_decoder
            })
            if not contains_all(network_info, ['recognizer_encoder', 'recognizer_decoder']):
                raise ConfigError('network_info should contain encoder and decoder fields')
        self.recognizer_encoder = create_recognizer(
            network_info['recognizer_encoder'], launcher, 'encoder', delayed_model_loading=delayed_model_loading)
        self.recognizer_decoder = create_recognizer(
            network_info['recognizer_decoder'], launcher, 'decoder', delayed_model_loading=delayed_model_loading)
        self.sos_index = 0
        self.eos_index = 2
        self.max_seq_len = int(network_info['max_seq_len'])
        self._part_by_name = {
            'encoder': self.recognizer_encoder,
            'decoder': self.recognizer_decoder
        }
        self.with_prefix = False

    def get_phrase(self, indices):
        raise NotImplementedError()

    def predict(self, identifiers, input_data, callback=None):
        raise NotImplementedError()

    def reset(self):
        pass

    def release(self):
        self.recognizer_encoder.release()
        self.recognizer_decoder.release()

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
            {'name': 'encoder', 'model': self.recognizer_encoder.network},
            {'name': 'decoder', 'model': self.recognizer_decoder.network}
        ]

    def update_inputs_outputs_info(self):
        def generate_name(prefix, with_prefix, layer_name):
            return prefix + layer_name if with_prefix else layer_name.split(prefix)[-1]

        with_prefix = next(iter(self.recognizer_encoder.network.input_info)).startswith('encoder')
        if with_prefix != self.with_prefix:
            for input_k, input_name in self.recognizer_encoder.inputs_mapping.items():
                self.recognizer_encoder.inputs_mapping[input_k] = generate_name('encoder_', with_prefix, input_name)
            for out_k, out_name in self.recognizer_encoder.outputs_mapping.items():
                self.recognizer_encoder.outputs_mapping[out_k] = generate_name('encoder_', with_prefix, out_name)
            for input_k, input_name in self.recognizer_decoder.inputs_mapping.items():
                self.recognizer_decoder.inputs_mapping[input_k] = generate_name('decoder_', with_prefix, input_name)
            for out_k, out_name in self.recognizer_decoder.outputs_mapping.items():
                self.recognizer_decoder.outputs_mapping[out_k] = generate_name('decoder_', with_prefix, out_name)
        self.with_prefix = with_prefix


class SequentialTextRecognitionModel(BaseSequentialModel):
    def __init__(self, network_info, launcher, models_args, meta, is_blob=None, delayed_model_loading=False):
        super().__init__(
            network_info, launcher, models_args, meta, is_blob=is_blob,
            delayed_model_loading=delayed_model_loading
        )
        self.vocab = network_info['custom_label_map']
        self.recognizer_encoder.inputs_mapping = {'imgs': 'imgs'}
        self.recognizer_encoder.outputs_mapping = {'features': 'features', 'decoder_hidden': 'decoder_hidden'}
        self.recognizer_decoder.inputs_mapping = {
            'features': 'features', 'hidden': 'hidden', 'decoder_input': 'decoder_input'
        }
        self.recognizer_decoder.outputs_mapping = {
            'decoder_hidden': 'decoder_hidden',
            'decoder_output': 'decoder_output'
        }

    def get_phrase(self, indices):
        res = ''.join(self.vocab.get(idx, '?') for idx in indices)
        return res

    def predict(self, identifiers, input_data, callback=None):
        assert len(identifiers) == 1
        input_data = np.array(input_data)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        enc_res = self.recognizer_encoder.predict(
            inputs={self.recognizer_encoder.inputs_mapping['imgs']: input_data})
        if callback:
            callback(enc_res)
        features = enc_res[self.recognizer_encoder.outputs_mapping['features']]
        dec_state = enc_res[self.recognizer_encoder.outputs_mapping['decoder_hidden']]

        tgt = np.array([[self.sos_index]])
        logits = []
        for _ in range(self.max_seq_len):

            dec_res = self.recognizer_decoder.predict(inputs={
                self.recognizer_decoder.inputs_mapping['features']: features,
                self.recognizer_decoder.inputs_mapping['hidden']: dec_state,
                self.recognizer_decoder.inputs_mapping['decoder_input']: tgt,
            })

            dec_state = dec_res[self.recognizer_decoder.outputs_mapping['decoder_hidden']]
            logit = dec_res[self.recognizer_decoder.outputs_mapping['decoder_output']]
            tgt = np.argmax(logit, axis=1)
            if self.eos_index == tgt[0]:
                break
            logits.append(logit)
            if callback:
                callback(dec_res)

        logits = np.array(logits)
        logits = logits.squeeze(axis=1)
        targets = np.argmax(logits, axis=1)
        result_phrase = self.get_phrase(targets)
        return result_phrase, dec_res


class SequentialFormulaRecognitionModel(BaseSequentialModel):
    def __init__(self, network_info, launcher, models_args, meta, is_blob=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, models_args, meta, is_blob,
                         delayed_model_loading=delayed_model_loading)
        self.vocab = meta.get('vocab')
        self.recognizer_encoder.inputs_mapping = {
            'imgs': 'imgs'
        }
        self.recognizer_encoder.outputs_mapping = {
            'row_enc_out': 'row_enc_out',
            'hidden': 'hidden',
            'context': 'context',
            'init_0': 'init_0'
        }
        self.recognizer_decoder.inputs_mapping = {
            'row_enc_out': 'row_enc_out',
            'dec_st_c': 'dec_st_c',
            'dec_st_h': 'dec_st_h',
            'output_prev': 'output_prev',
            'tgt': 'tgt'
        }
        self.recognizer_decoder.outputs_mapping = {
            'dec_st_h_t': 'dec_st_h_t',
            'dec_st_c_t': 'dec_st_c_t',
            'output': 'output',
            'logit': 'logit'
        }

    def get_phrase(self, indices):
        res = ''
        for idx in indices:
            if idx != self.eos_index:
                res += ' ' + str(self.vocab.get(idx, '?'))
            else:
                return res.strip()
        return res.strip()

    def predict(self, identifiers, input_data, callback=None):
        assert len(identifiers) == 1
        input_data = np.array(input_data)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        enc_res = self.recognizer_encoder.predict(
            inputs={self.recognizer_encoder.inputs_mapping['imgs']: input_data})
        if callback:
            callback(enc_res)
        row_enc_out = enc_res[self.recognizer_encoder.outputs_mapping['row_enc_out']]
        dec_states_h = enc_res[self.recognizer_encoder.outputs_mapping['hidden']]
        dec_states_c = enc_res[self.recognizer_encoder.outputs_mapping['context']]
        O_t = enc_res[self.recognizer_encoder.outputs_mapping['init_0']]

        tgt = np.array([[self.sos_index]])
        logits = []
        for _ in range(self.max_seq_len):

            dec_res = self.recognizer_decoder.predict(inputs={
                self.recognizer_decoder.inputs_mapping['row_enc_out']: row_enc_out,
                self.recognizer_decoder.inputs_mapping['dec_st_c']: dec_states_c,
                self.recognizer_decoder.inputs_mapping['dec_st_h']: dec_states_h,
                self.recognizer_decoder.inputs_mapping['output_prev']: O_t,
                self.recognizer_decoder.inputs_mapping['tgt']: tgt
            })
            if callback:
                callback(dec_res)

            dec_states_h = dec_res[self.recognizer_decoder.outputs_mapping['dec_st_h_t']]
            dec_states_c = dec_res[self.recognizer_decoder.outputs_mapping['dec_st_c_t']]
            O_t = dec_res[self.recognizer_decoder.outputs_mapping['output']]
            logit = dec_res[self.recognizer_decoder.outputs_mapping['logit']]
            logits.append(logit)
            tgt = np.array([[np.argmax(np.array(logit), axis=1)]])

            if tgt[0][0][0] == self.eos_index:
                break

        logits = np.array(logits)
        logits = logits.squeeze(axis=1)
        targets = np.argmax(logits, axis=1)
        result_phrase = self.get_phrase(targets)
        return result_phrase, dec_res


class RecognizerDLSDKModel(BaseModel):
    def __init__(self, network_info, launcher, suffix,
                 delayed_model_loading=False, inputs_mapping=None, outputs_mapping=None):
        super().__init__(network_info, launcher, suffix)
        self.launcher = launcher
        self.is_dynamic = False
        if not delayed_model_loading:
            model, weights = self.automatic_model_search(network_info)
            if weights is not None:
                self.network = launcher.read_network(str(model), str(weights))
                self.load_network(self.network, launcher)
            else:
                self.exec_network = launcher.ie_core.import_network(str(model))
            self.print_input_output_info()
        self.inputs_mapping = inputs_mapping
        self.outputs_mapping = outputs_mapping

    def predict(self, inputs, identifiers=None):
        if not self.is_dynamic and self.dynamic_inputs:
            self.reshape_net({k: v.shape for k, v in inputs.items()})
        return self.exec_network.infer(inputs)

    def release(self):
        del self.exec_network

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.load_network(self.network, launcher)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        if log:
            self.print_input_output_info()

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
        if not self.dynamic_inputs:
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)

    def reshape_net(self, shape):
        if self.is_dynamic:
            return
        if hasattr(self, 'exec_network') and self.exec_network is not None:
            del self.exec_network
        self.network.reshape(shape)
        self.dynamic_inputs, self.partial_shapes = self.launcher.get_dynamic_inputs(self.network)
        if not self.is_dynamic and self.dynamic_inputs:
            return
        self.exec_network = self.launcher.load_network(self.network, self.launcher.device)

    def get_network(self):
        return self.network


MODEL_TYPES = {
    'SequentialTextRecognitionModel': SequentialTextRecognitionModel,
    'SequentialFormulaRecognitionModel': SequentialFormulaRecognitionModel,
}
