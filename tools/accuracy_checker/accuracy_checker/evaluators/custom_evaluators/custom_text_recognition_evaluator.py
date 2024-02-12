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
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseDLSDKModel, BaseOpenVINOModel, BaseCascadeModel, create_model
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, generate_layer_name, postprocess_output_name
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
        meta = {}
        if config.get('custom_label_map'):
            meta.update({
                'custom_label_map': config['custom_label_map']
            })
        if config.get('max_seq_len'):
            meta.update({
                'max_seq_len': config['max_seq_len']
            })
        model = MODEL_TYPES[model_type](
            config.get('network_info', {}), launcher, config.get('_models', []), meta, config.get('_model_is_blob'),
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
            self.model.vocab = self.dataset.metadata.get('vocab', {})


class BaseSequentialModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, meta, is_blob=None, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['recognizer_encoder', 'recognizer_decoder']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder and decoder fields')
        self._recognizer_mapping = {
            'dlsdk': RecognizerDLSDKModel,
            'openvino': RecognizerOVModel,
        }
        self.recognizer_encoder = create_model(network_info['recognizer_encoder'], launcher, self._recognizer_mapping,
                                               'encoder', delayed_model_loading=delayed_model_loading)
        self.recognizer_decoder = create_model(network_info['recognizer_decoder'], launcher, self._recognizer_mapping,
                                               'decoder', delayed_model_loading=delayed_model_loading)
        self.sos_index = 0
        self.eos_index = 2
        self.max_seq_len = int(meta.get('max_seq_len', 0))
        self._part_by_name = {'encoder': self.recognizer_encoder, 'decoder': self.recognizer_decoder}
        self.with_prefix = False

    def load_model(self, network_list, launcher):
        super().load_model(network_list, launcher)
        self.update_inputs_outputs_info()

    def load_network(self, network_list, launcher):
        super().load_network(network_list, launcher)
        self.update_inputs_outputs_info()

    def update_inputs_outputs_info(self):
        if not hasattr(self.recognizer_encoder, 'inputs'):
            return
        with_prefix = next(iter(self.recognizer_encoder.inputs)).startswith('encoder')
        if with_prefix != self.with_prefix:
            for input_k, input_name in self.recognizer_encoder.inputs_mapping.items():
                self.recognizer_encoder.inputs_mapping[input_k] = generate_layer_name(input_name, 'encoder_',
                                                                                      with_prefix)
            for input_k, input_name in self.recognizer_decoder.inputs_mapping.items():
                self.recognizer_decoder.inputs_mapping[input_k] = generate_layer_name(input_name, 'decoder_',
                                                                                      with_prefix)
        self.with_prefix = with_prefix
        if hasattr(self.recognizer_encoder, 'outputs'):
            outputs_mapping = self.recognizer_encoder.outputs
            for output_k in self.recognizer_encoder.outputs_mapping:
                postprocessed_name = postprocess_output_name(
                    self.recognizer_encoder.outputs_mapping[output_k], outputs_mapping,
                    additional_mapping=self.recognizer_encoder.additional_output_mapping, raise_error=False
                )
                if postprocessed_name not in outputs_mapping:
                    postprocessed_name = postprocess_output_name(
                    generate_layer_name(self.recognizer_encoder.outputs_mapping[output_k], 'encoder_', with_prefix),
                    outputs_mapping,
                    additional_mapping=self.recognizer_encoder.additional_output_mapping, raise_error=False
                )
                self.recognizer_encoder.outputs_mapping[output_k] = postprocessed_name

            outputs_mapping = self.recognizer_decoder.outputs
            for output_k in self.recognizer_decoder.outputs_mapping:
                postprocessed_name = postprocess_output_name(
                    self.recognizer_decoder.outputs_mapping[output_k], outputs_mapping,
                    additional_mapping=self.recognizer_decoder.additional_output_mapping, raise_error=False
                )
                if postprocessed_name not in outputs_mapping:
                    postprocessed_name = postprocess_output_name(
                    generate_layer_name(self.recognizer_decoder.outputs_mapping[output_k], 'decoder_', with_prefix),
                    outputs_mapping,
                    additional_mapping=self.recognizer_decoder.additional_output_mapping, raise_error=False
                )
                self.recognizer_decoder.outputs_mapping[output_k] = postprocessed_name

    def predict(self, identifiers, input_data):
        pass


class SequentialTextRecognitionModel(BaseSequentialModel):
    def __init__(self, network_info, launcher, models_args, meta, is_blob=None, delayed_model_loading=False):
        super().__init__(
            network_info, launcher, models_args, meta, is_blob=is_blob,
            delayed_model_loading=delayed_model_loading
        )
        self.vocab = meta.get('custom_label_map')
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
        enc_res = self.recognizer_encoder.predict(identifiers,
                                                  {self.recognizer_encoder.inputs_mapping['imgs']: input_data})
        if isinstance(enc_res, tuple):
            enc_res, enc_raw_res = enc_res
        else:
            enc_raw_res = enc_res

        if callback:
            callback(enc_raw_res)
        feats_out = postprocess_output_name(self.recognizer_encoder.outputs_mapping['features'], enc_res,
                                            additional_mapping=self.recognizer_encoder.additional_output_mapping)
        hidden_out = postprocess_output_name(self.recognizer_encoder.outputs_mapping['decoder_hidden'], enc_res,
                                             additional_mapping=self.recognizer_encoder.additional_output_mapping)
        features = enc_res[feats_out]
        dec_state = enc_res[hidden_out]

        tgt = np.array([self.sos_index])
        logits = []
        for _ in range(self.max_seq_len):

            dec_res = self.recognizer_decoder.predict(
                identifiers,
                {
                    self.recognizer_decoder.inputs_mapping['features']: features,
                    self.recognizer_decoder.inputs_mapping['hidden']: dec_state,
                    self.recognizer_decoder.inputs_mapping['decoder_input']: tgt
                 })
            if isinstance(dec_res, tuple):
                dec_res, dec_raw_res = dec_res
            else:
                dec_raw_res = dec_res

            logits_out = postprocess_output_name(self.recognizer_decoder.outputs_mapping['decoder_output'], dec_res,
                                                 additional_mapping=self.recognizer_decoder.additional_output_mapping)
            hidden_out = postprocess_output_name(self.recognizer_decoder.outputs_mapping['decoder_hidden'], dec_res,
                                                 additional_mapping=self.recognizer_decoder.additional_output_mapping)
            dec_state = dec_res[hidden_out]
            logit = dec_res[logits_out]
            tgt = np.argmax(logit, axis=1)
            if self.eos_index == tgt[0]:
                break
            logits.append(logit)
            if callback:
                callback(dec_raw_res)

        logits = np.array(logits)
        logits = logits.squeeze(axis=1)
        targets = np.argmax(logits, axis=1)
        result_phrase = self.get_phrase(targets)
        return result_phrase, dec_raw_res


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
        self.update_inputs_outputs_info()

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
        enc_res = self.recognizer_encoder.predict(identifiers,
                                                  {self.recognizer_encoder.inputs_mapping['imgs']: input_data})
        if isinstance(enc_res, tuple):
            enc_res, enc_raw_res = enc_res
        else:
            dec_raw_res = enc_res
        if callback:
            callback(enc_raw_res)
        row_enc_out = enc_res[self.recognizer_encoder.outputs_mapping['row_enc_out']]
        dec_states_h = enc_res[self.recognizer_encoder.outputs_mapping['hidden']]
        dec_states_c = enc_res[self.recognizer_encoder.outputs_mapping['context']]
        O_t = enc_res[self.recognizer_encoder.outputs_mapping['init_0']]

        tgt = np.array([[self.sos_index]])
        logits = []
        for _ in range(self.max_seq_len):

            dec_res = self.recognizer_decoder.predict(
                identifiers,
                {
                    self.recognizer_decoder.inputs_mapping['row_enc_out']: row_enc_out,
                    self.recognizer_decoder.inputs_mapping['dec_st_c']: dec_states_c,
                    self.recognizer_decoder.inputs_mapping['dec_st_h']: dec_states_h,
                    self.recognizer_decoder.inputs_mapping['output_prev']: O_t,
                    self.recognizer_decoder.inputs_mapping['tgt']: tgt
                })
            if isinstance(dec_res, tuple):
                dec_res, dec_raw_res = dec_res
            else:
                dec_raw_res = dec_res
            if callback:
                callback(dec_raw_res)

            dec_states_h = dec_res[self.recognizer_decoder.outputs_mapping['dec_st_h_t']]
            dec_states_c = dec_res[self.recognizer_decoder.outputs_mapping['dec_st_c_t']]
            O_t = dec_res[self.recognizer_decoder.outputs_mapping['output']]
            logit = dec_res[self.recognizer_decoder.outputs_mapping['logit']]
            logits.append(logit)
            tgt = np.array([np.argmax(np.array(logit), axis=1)])

            if tgt[0][0] == self.eos_index:
                break

        logits = np.array(logits)
        logits = logits.squeeze(axis=1)
        targets = np.argmax(logits, axis=1)
        result_phrase = self.get_phrase(targets)
        return result_phrase, dec_raw_res


class RecognizerDLSDKModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix,
                 delayed_model_loading=False, inputs_mapping=None, outputs_mapping=None):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.inputs_mapping = inputs_mapping
        self.outputs_mapping = outputs_mapping

    def predict(self, identifiers, input_data):
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in input_data.items()})
        return self.exec_network.infer(input_data)


class RecognizerOVModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix,
                 delayed_model_loading=False, inputs_mapping=None, outputs_mapping=None):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.inputs_mapping = inputs_mapping
        self.outputs_mapping = outputs_mapping

    def predict(self, identifiers, input_data):
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in input_data.items()})
        return self.infer(input_data, raw_results=True)


MODEL_TYPES = {
    'SequentialTextRecognitionModel': SequentialTextRecognitionModel,
    'SequentialFormulaRecognitionModel': SequentialFormulaRecognitionModel,
}
