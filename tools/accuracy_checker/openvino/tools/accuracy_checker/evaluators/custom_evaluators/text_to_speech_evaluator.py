"""
Copyright (c) 2018-2022 Intel Corporation

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
from .base_models import BaseCascadeModel, BaseDLSDKModel, create_model, BaseOpenVINOModel
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, generate_layer_name, postprocess_output_name


class TextToSpeechEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        adapter_info = config['adapter']
        pos_mask_window = config['pos_mask_window']
        model = SequentialModel(
            config.get('network_info', {}), launcher, config.get('_models', []), adapter_info, pos_mask_window,
            config.get('_model_is_blob'), delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_data, batch_meta = extract_image_representations(batch_inputs)
            input_names = ['{}{}'.format(
                'forward_tacotron_duration_' if self.model.with_prefix else '',
                s.split('.')[-1]) for s in batch_inputs[0].identifier]
            temporal_output_callback = None
            if output_callback:
                temporal_output_callback = partial(output_callback, metrics_result=None,
                                                   element_identifiers=batch_identifiers,
                                                   dataset_indices=batch_input_ids)
            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_data, batch_meta, input_names, callback=temporal_output_callback
            )
            batch_annotation, batch_prediction = self.postprocessor.process_batch(batch_annotation, batch_prediction)
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


class SequentialModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, adapter_info, pos_mask_window, is_blob=None,
                 delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['forward_tacotron_duration', 'forward_tacotron_regression', 'melgan']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain forward_tacotron_duration,'
                              'forward_tacotron_regression and melgan fields')
        self._duration_mapping = {
            'dlsdk': TTSDLSDKModel,
            'openvino': TTSOVModel
        }
        self._regression_mapping = {
            'dlsdk': RegressionDLSDKModel,
            'openvino': RegressionOVModel
        }
        self._melgan_mapping = {
            'dlsdk': MelganDLSDKModel,
            'openvino': MelganOVModel
        }
        self.forward_tacotron_duration = create_model(
            network_info.get('forward_tacotron_duration', {}), launcher, self._duration_mapping,
            'duration_prediction_att', delayed_model_loading
        )
        self.forward_tacotron_regression = create_model(
            network_info.get('forward_tacotron_regression', {}), launcher, self._regression_mapping,
            'regression_att', delayed_model_loading
        )
        self.melgan = create_model(
            network_info.get('melgan', {}), launcher, self._melgan_mapping, "melganupsample", delayed_model_loading
        )
        if not delayed_model_loading:
            self.forward_tacotron_duration_input = next(iter(self.forward_tacotron_duration.inputs))
            self.melgan_input = next(iter(self.melgan.inputs))
        else:
            self.forward_tacotron_duration_input = None
            self.melgan_input = None
        self.duration_speaker_embeddings = (
            'speaker_embedding' if 'speaker_embedding' in self.forward_tacotron_regression_input else None
        )
        self.duration_output = 'duration'
        self.embeddings_output = 'embeddings'
        self.mel_output = 'mel'
        self.audio_output = 'audio'
        self.pos_mask_window = int(pos_mask_window)
        self.adapter = create_adapter(adapter_info)
        self.adapter.output_blob = self.audio_output
        if not delayed_model_loading:
            self.update_inputs_outputs_info()

        self.init_pos_mask(window_size=self.pos_mask_window)

        self.with_prefix = False
        self._part_by_name = {
            'forward_tacotron_duration': self.forward_tacotron_duration,
            'forward_tacotron_regression': self.forward_tacotron_regression,
            'melgan': self.melgan
        }

    @property
    def forward_tacotron_regression_input(self):
        return self.forward_tacotron_regression.regression_input

    @property
    def max_mel_len(self):
        return self.melgan.max_len

    @property
    def max_regression_len(self):
        return self.forward_tacotron_regression.max_len

    def init_pos_mask(self, mask_sz=6000, window_size=4):
        mask_arr = np.zeros((1, 1, mask_sz, mask_sz), dtype=np.float32)
        width = 2 * window_size + 1
        for i in range(mask_sz - width):
            mask_arr[0][0][i][i:i + width] = 1.0

        self.pos_mask = mask_arr

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = np.max(length)
        x = np.arange(max_length, dtype=length.dtype)
        x = np.expand_dims(x, axis=(0))
        length = np.expand_dims(length, axis=(1))
        return x < length

    def predict(self, identifiers, input_data, input_meta=None, input_names=None, callback=None):
        assert len(identifiers) == 1

        duration_input = dict(zip(input_names, input_data[0]))
        duration_output = self.forward_tacotron_duration.predict(identifiers, duration_input)
        if isinstance(duration_output, tuple):
            duration_output, raw_duration_output = duration_output
        else:
            raw_duration_output = duration_output

        if callback:
            callback(raw_duration_output)

        duration = duration_output[self.duration_output]
        duration = (duration + 0.5).astype('int').flatten()
        duration = np.expand_dims(duration, axis=0)

        preprocessed_emb = duration_output[self.embeddings_output]
        indexes = self.build_index(duration, preprocessed_emb)
        processed_emb = self.gather(preprocessed_emb, 1, indexes)
        processed_emb = processed_emb[:, :self.max_regression_len, :]
        if len(input_names) > 1:  # in the case of network with attention
            input_mask = self.sequence_mask(np.array([[processed_emb.shape[1]]]), processed_emb.shape[1])
            pos_mask = self.pos_mask[:, :, :processed_emb.shape[1], :processed_emb.shape[1]]
            input_to_regression = {
                self.forward_tacotron_regression_input['data']: processed_emb,
                self.forward_tacotron_regression_input['data_mask']: input_mask,
                self.forward_tacotron_regression_input['pos_mask']: pos_mask}
            if self.duration_speaker_embeddings:
                sp_emb_input = self.forward_tacotron_regression_input['speaker_embedding']
                input_to_regression[sp_emb_input] = duration_input[self.duration_speaker_embeddings]
            mels = self.forward_tacotron_regression.predict(identifiers, input_to_regression)
        else:
            mels = self.forward_tacotron_regression.predict(identifiers,
                                                            {self.forward_tacotron_regression_input: processed_emb})
        if isinstance(mels, tuple):
            mels, raw_mels = mels
        else:
            raw_mels = mels
        if callback:
            callback(raw_mels)
        melgan_input = mels[self.mel_output]
        if np.ndim(melgan_input) != 3:
            melgan_input = np.expand_dims(melgan_input, 0)
        melgan_input = melgan_input[:, :, :self.max_mel_len]
        audio = self.melgan.predict(identifiers, {self.melgan_input: melgan_input})
        if isinstance(audio, tuple):
            audio, raw_audio = audio
        else:
            raw_audio = audio

        return raw_audio, self.adapter.process(audio, identifiers, input_meta)

    def load_model(self, network_list, launcher):
        super().load_model(network_list, launcher)
        self.update_inputs_outputs_info()

    def load_network(self, network_list, launcher):
        super().load_network(network_list, launcher)
        self.update_inputs_outputs_info()

    @staticmethod
    def build_index(duration, x):
        duration[np.where(duration < 0)] = 0
        tot_duration = np.cumsum(duration, 1)
        max_duration = int(tot_duration.max().item())
        index = np.zeros([x.shape[0], max_duration, x.shape[2]], dtype='long')

        for i in range(tot_duration.shape[0]):
            pos = 0
            for j in range(tot_duration.shape[1]):
                pos1 = tot_duration[i, j]
                index[i, pos:pos1, :] = j
                pos = pos1
            index[i, pos:, :] = tot_duration.shape[1] - 1
        return index

    @staticmethod
    def gather(a, dim, index):
        expanded_index = [
            index if dim == i else
            np.arange(a.shape[i]).reshape([-1 if i == j else 1 for j in range(a.ndim)]) for i in range(a.ndim)
        ]
        return a[tuple(expanded_index)]

    def update_inputs_outputs_info(self):
        if hasattr(self.forward_tacotron_duration, 'outputs'):
            self.duration_output = postprocess_output_name(
                self.duration_output, self.forward_tacotron_duration.outputs, raise_error=False)
            self.embeddings_output = postprocess_output_name(
                self.embeddings_output, self.forward_tacotron_duration.outputs, raise_error=False)
            self.mel_output = postprocess_output_name(self.mel_output, self.forward_tacotron_regression.outputs,
                                                      raise_error=False)
            self.audio_output = postprocess_output_name(self.audio_output, self.melgan.outputs, raise_error=False)
            self.adapter.output_blob = self.audio_output
        current_name = next(iter(self.forward_tacotron_duration.inputs))
        with_prefix = current_name.startswith('forward_tacotron_duration_')
        if not hasattr(self, 'with_prefix') or with_prefix != self.with_prefix:
            self.forward_tacotron_duration_input = next(iter(self.forward_tacotron_duration.inputs))
            self.melgan_input = next(iter(self.melgan.inputs))
            if self.duration_speaker_embeddings:
                self.duration_speaker_embeddings = generate_layer_name(
                    self.duration_speaker_embeddings, 'forward_tacotron_duration_', with_prefix
                )
            for key, value in self.forward_tacotron_regression_input.items():
                self.forward_tacotron_regression_input[key] = generate_layer_name(
                    value, 'forward_tacotron_regression_', with_prefix
                )

        self.with_prefix = with_prefix


class TextToSpeechEvaluatorGan(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        adapter_info = config['adapter']

        model = SequentialModelGAN(
            config.get('network_info', {}), launcher, config.get('_models', []), adapter_info,
            config.get('_model_is_blob'), delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_data, batch_meta = extract_image_representations(batch_inputs)
            input_names = ['{}{}'.format(s.split('.')[-1]) for s in batch_inputs[0].identifier]
            temporal_output_callback = None
            if output_callback:
                temporal_output_callback = partial(output_callback, metrics_result=None,
                                                   element_identifiers=batch_identifiers,
                                                   dataset_indices=batch_input_ids)
            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_data, batch_meta, input_names, callback=temporal_output_callback
            )
            batch_annotation, batch_prediction = self.postprocessor.process_batch(batch_annotation, batch_prediction)
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


class SequentialModelGAN(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, adapter_info, is_blob=None,
                 delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['encoder', 'decoder']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain forward_tacotron_duration,'
                              'forward_tacotron_regression and melgan fields')
        self._encoder_mapping = {
            'dlsdk': TTSDLSDKModel,
            'openvino': TTSOVModel
        }
        self._decoder_mapping = {
            'dlsdk': RegressionDLSDKModel,
            'openvino': RegressionOVModel
        }

        self.encoder = create_model(
            network_info.get('encoder', {}), launcher, self._encoder_mapping,
            'encoder', delayed_model_loading
        )
        self.decoder = create_model(
            network_info.get('decoder', {}), launcher, self._decoder_mapping,
            'encoder', delayed_model_loading
        )

        if not delayed_model_loading:
            self.encoder_input = next(iter(self.encoder.inputs))
        else:
            self.encoder_input = None

        self.encoder_output_for_decoder = 'x_res'
        self.encoder_output_for_attention = 'x_m'
        self.encoder_output_for_duration = 'logw'
        self.encoder_output_mask = 'x_mask'

        self.decoder_output = 'mel'

        self.adapter = create_adapter(adapter_info)
        self.adapter.output_blob = self.decoder_output

        self.with_prefix = False
        self._part_by_name = {
            'encoder': self.encoder,
            'decoder': self.decoder,
        }

    @property
    def decoder_input(self):
        return self.decoder.regression_input

    @staticmethod
    def compute_train_attention_map(x_m: np.ndarray, mel: np.ndarray, attn_mask: np.ndarray) -> np.ndarray:
        mul = np.ones_like(x_m)  # [b, d, t]
        x_2 = np.sum(-x_m ** 2, 1)  # np.expand_dims(np.sum(-x_m ** 2, 1), axis=-1)  # [b, t, 1]
        z_2 = np.matmul(mul.transpose((0, 2, 1)), -mel ** 2)  # [b, t, t']
        xz2 = np.matmul(x_m.transpose((0, 2, 1)), mel)  # [b, t, d] * [b, d, t'] = [b, t, t']

        corr_coeff = z_2 + x_2[:, :, np.newaxis]
        corr_coeff = corr_coeff + 2 * xz2

        attn = SequentialModelGAN.maximum_path_np(corr_coeff, np.squeeze(attn_mask, 1)).astype(x_m.dtype)
        return attn

    @staticmethod
    def maximum_path_np(map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        map: [b, t_text, t_mel]
        mask: [b, t_text, t_mel]
        """
        map = map * mask

        path = np.zeros_like(map).astype(np.int32)

        t_text_max = mask.sum(1)[:, 0].astype(np.int32)
        t_mel_max = mask.sum(2)[:, 0].astype(np.int32)

        for b in range(map.shape[0]):
            min_val = -1e9
            t_text, t_mel = t_text_max[b], t_mel_max[b]
            index = t_text - 1

            for x in range(t_mel):
                for y in range(max(0, t_text + x - t_mel), min(t_text, x + 1)):
                    if x == y:
                        v_cur = min_val
                    else:
                        v_cur = map[b, y, x - 1]
                    if y == 0:
                        if x == 0:
                            v_prev = 0.
                        else:
                            v_prev = min_val
                    else:
                        v_prev = map[b, y - 1, x - 1]
                    map[b, y, x] = max(v_cur, v_prev) + map[b, y, x]

            for x in range(t_mel - 1, -1, -1):
                path[b, index, x] = 1
                if index != 0 and (index == x or map[b, index, x - 1] < map[b, index - 1, x - 1]):
                    index = index - 1

        return path

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = np.max(length)
        x = np.arange(max_length, dtype=length.dtype)
        x = np.expand_dims(x, axis=(0))
        length = np.expand_dims(length, axis=(1))
        return x < length

    def predict(self, identifiers, input_data, input_meta=None, input_names=None, callback=None):
        assert len(identifiers) == 1

        encoder_input = dict(zip(input_names, input_data[0]))
        encoder_output = self.encoder.predict(identifiers, encoder_input)
        if isinstance(encoder_output, tuple):
            encoder_output, raw_encoder_output = encoder_output
        else:
            raw_encoder_output = encoder_output

        if callback:
            callback(raw_encoder_output)

        x_mask = encoder_output[self.encoder_output_mask]
        x_m = encoder_output[self.encoder_output_for_attention]
        x_res = encoder_output[self.encoder_output_for_decoder]
        # next line could be used for comparison between predicted duration and real one
        #logw = encoder_output[self.encoder_output_for_duration]

        mel = input_data[1]
        mel_max_length = mel.shape[-1]

        z_mask = SequentialModelGAN.sequence_mask(np.array([mel_max_length]), mel_max_length)
        z_mask = np.expand_dims(z_mask, axis=[1, 2])
        x_mask = np.expand_dims(x_mask, axis=[-1])
        attn_mask = x_mask * z_mask
        z_mask = np.squeeze(z_mask, 2)

        attn = SequentialModelGAN.compute_train_attention_map(x_m, mel, attn_mask)
        z = np.matmul(attn.transpose((0, 2, 1)), x_res.transpose((0, 2, 1))).transpose((0, 2, 1))

        input_to_decoder = {
            self.decoder_input['z']: z,
            self.decoder_input['z_mask']: z_mask}

        mels = self.decoder.predict(identifiers, input_to_decoder)

        if isinstance(mels, tuple):
            mels, raw_mels = mels
        else:
            raw_mels = mels
        if callback:
            callback(raw_mels)

        return raw_mels, self.adapter.process(mels, identifiers, input_meta)

    def load_model(self, network_list, launcher):
        super().load_model(network_list, launcher)
        self.update_inputs_outputs_info()

    def load_network(self, network_list, launcher):
        super().load_network(network_list, launcher)
        self.update_inputs_outputs_info()

    def update_inputs_outputs_info(self):
        if hasattr(self.encoder, 'outputs'):
            self.encoder_output_for_decoder = postprocess_output_name(
                self.encoder_output_for_decoder, self.encoder.outputs, raise_error=False)
            self.encoder_output_for_attention = postprocess_output_name(
                self.encoder_output_for_attention, self.encoder.outputs, raise_error=False)
            self.encoder_output_for_duration = postprocess_output_name(
                self.encoder_output_for_duration, self.encoder.outputs, raise_error=False)
            self.encoder_output_mask = postprocess_output_name(
                self.encoder_output_mask, self.encoder.outputs, raise_error=False)

            self.decoder_output = postprocess_output_name(
                self.decoder_output, self.decoder.outputs, raise_error=False)
            self.adapter.output_blob = self.decoder_output

class TTSDLSDKModel(BaseDLSDKModel):
    def predict(self, identifiers, input_data):
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in input_data.items()})
        return self.exec_network.infer(input_data)

    @property
    def inputs(self):
        if self.network:
            return self.network.input_info if hasattr(self.network, 'input_info') else self.network.inputs
        return self.exec_network.input_info if hasattr(self.exec_network, 'input_info') else self.exec_network.inputs


class TTSOVModel(BaseOpenVINOModel):
    def predict(self, identifiers, input_data):
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in input_data.items()})
        return self.infer(input_data)

    def infer(self, input_data, raw_results=True):
        return super().infer(input_data, raw_results)

    def set_input_and_output(self):
        pass


class RegressionDLSDKModel(TTSDLSDKModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.max_len = int(network_info['max_regression_len'])
        self.regression_input = network_info['inputs']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class MelganDLSDKModel(TTSDLSDKModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.max_len = int(network_info['max_mel_len'])
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class RegressionOVModel(TTSOVModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.max_len = int(network_info['max_regression_len'])
        self.regression_input = network_info['inputs']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class MelganOVModel(TTSOVModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.max_len = int(network_info['max_mel_len'])
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
