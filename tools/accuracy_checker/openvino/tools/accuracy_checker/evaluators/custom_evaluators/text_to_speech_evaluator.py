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
from functools import partial
from collections import OrderedDict
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, get_path
from ...logging import print_info


class TextToSpeechEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = SequentialModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
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


def create_network(model_config, launcher, suffix, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': TTSDLSDKModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, suffix, delayed_model_loading)


class SequentialModel:
    def __init__(self, network_info, launcher, models_args, is_blob=None, delayed_model_loading=False):
        if not delayed_model_loading:
            forward_tacotron_duration = network_info.get('forward_tacotron_duration', {})
            forward_tacotron_regression = network_info.get('forward_tacotron_regression', {})
            melgan = network_info.get('melgan', {})
            if 'model' not in forward_tacotron_duration:
                forward_tacotron_duration['model'] = models_args[0]
                forward_tacotron_duration['_model_is_blob'] = is_blob
            if 'model' not in forward_tacotron_regression:
                forward_tacotron_regression['model'] = models_args[1 if len(models_args) > 1 else 0]
                forward_tacotron_regression['_model_is_blob'] = is_blob
            if 'model' not in melgan:
                melgan['model'] = models_args[2 if len(models_args) > 2 else 0]
                melgan['_model_is_blob'] = is_blob
            network_info.update({
                'forward_tacotron_duration': forward_tacotron_duration,
                'forward_tacotron_regression': forward_tacotron_regression,
                'melgan': melgan
            })
            required_fields = ['forward_tacotron_duration', 'forward_tacotron_regression', 'melgan']
            if not contains_all(network_info, required_fields):
                raise ConfigError(
                    'network_info should contains: {} fields'.format(' ,'.join(required_fields))
                )
        self.forward_tacotron_duration = create_network(
            network_info.get('forward_tacotron_duration', {}), launcher,
            'duration_prediction_att', delayed_model_loading
        )
        self.forward_tacotron_regression = create_network(
            network_info.get('forward_tacotron_regression', {}), launcher,
            'regression_att', delayed_model_loading
        )
        self.melgan = create_network(
            network_info.get('melgan', {}), launcher, "melganupsample", delayed_model_loading
        )
        if not delayed_model_loading:
            self.forward_tacotron_duration_input = next(iter(self.forward_tacotron_duration.inputs))
            self.melgan_input = next(iter(self.melgan.inputs))
        else:
            self.forward_tacotron_duration_input = None
            self.melgan_input = None
        self.forward_tacotron_regression_input = network_info['forward_tacotron_regression_inputs']
        self.duration_speaker_embeddings = (
            'speaker_embedding' if 'speaker_embedding' in self.forward_tacotron_regression_input else None
        )
        self.duration_output = 'duration'
        self.embeddings_output = 'embeddings'
        self.mel_output = 'mel'
        self.audio_output = 'audio'
        self.max_mel_len = int(network_info['max_mel_len'])
        self.max_regression_len = int(network_info['max_regression_len'])
        self.pos_mask_window = int(network_info['pos_mask_window'])
        self.adapter = create_adapter(network_info['adapter'])
        self.adapter.output_blob = self.audio_output

        self.init_pos_mask(window_size=self.pos_mask_window)

        self.with_prefix = False
        self._part_by_name = {
            'forward_tacotron_duration': self.forward_tacotron_duration,
            'forward_tacotron_regression': self.forward_tacotron_regression,
            'melgan': self.melgan
        }

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

    def predict(self, identifiers, input_data, input_meta, input_names, callback=None):
        assert len(identifiers) == 1

        duration_input = dict(zip(input_names, input_data[0]))
        duration_output = self.forward_tacotron_duration.predict(duration_input)
        if callback:
            callback(duration_output)

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
            mels = self.forward_tacotron_regression.predict(input_to_regression)
        else:
            mels = self.forward_tacotron_regression.predict({self.forward_tacotron_regression_input: processed_emb})
        if callback:
            callback(mels)
        melgan_input = mels[self.mel_output]
        if np.ndim(melgan_input) != 3:
            melgan_input = np.expand_dims(melgan_input, 0)
        melgan_input = melgan_input[:, :, :self.max_mel_len]
        audio = self.melgan.predict({self.melgan_input: melgan_input})

        return audio, self.adapter.process(audio, identifiers, input_meta)

    def release(self):
        self.forward_tacotron_duration.release()
        self.forward_tacotron_regression.release()
        self.melgan.release()

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
            {'name': 'forward_tacotron_duration', 'model': self.forward_tacotron_duration.get_network()},
            {'name': 'forward_tacotron_regression', 'model': self.forward_tacotron_regression.get_network()},
            {'name': 'melgan', 'model': self.melgan.get_network()}
        ]

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
        def generate_name(prefix, with_prefix, layer_name):
            return prefix + layer_name if with_prefix else layer_name.split(prefix)[-1]

        current_name = next(iter(self.forward_tacotron_duration.inputs))
        with_prefix = current_name.startswith('forward_tacotron_duration_')
        if with_prefix != self.with_prefix:
            self.duration_output = generate_name('forward_tacotron_duration_', with_prefix, self.duration_output)
            self.embeddings_output = generate_name('forward_tacotron_duration_', with_prefix, self.embeddings_output)
            self.mel_output = generate_name('forward_tacotron_regression_', with_prefix, self.mel_output)
            self.audio_output = generate_name('melgan_', with_prefix, self.audio_output)
            self.adapter.output_blob = self.audio_output
            self.forward_tacotron_duration_input = next(iter(self.forward_tacotron_duration.inputs))
            self.melgan_input = next(iter(self.melgan.inputs))
            if self.duration_speaker_embeddings:
                self.duration_speaker_embeddings = generate_name(
                    'forward_tacotron_duration_', with_prefix, self.duration_speaker_embeddings
                )
            for key, value in self.forward_tacotron_regression_input.items():
                self.forward_tacotron_regression_input[key] = generate_name(
                    'forward_tacotron_regression_', with_prefix, value
                )

        self.with_prefix = with_prefix


class TTSDLSDKModel:
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.network_info = network_info
        self.default_model_suffix = suffix
        self.is_dynamic = False
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)
        self.launcher = launcher

    def predict(self, input_data):
        if not self.is_dynamic and self.dynamic_inputs:
            self.reshape({k: v.shape for k, v in input_data.items()})
        return self.exec_network.infer(input_data)

    def release(self):
        del self.network
        del self.exec_network

    def reshape(self, input_shapes):
        if not hasattr(self, 'is_dynamic'):
            self.is_dynamic = False
        if not self.is_dynamic:
            del self.exec_network
            self.network.reshape(input_shapes)
            self.dynamic_inputs, self.partial_shapes = self.launcher.get_dynamic_inputs(self.network)
            if not self.is_dynamic and self.dynamic_inputs:
                self.exec_network = None
                return
            self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)

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
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
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
            print_info('\tshape: {}\n'.format(
                input_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))
        print_info('{} - Output info'.format(self.default_model_suffix))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(
                output_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))

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

    def get_network(self):
        return self.network

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.load_network(self.network, launcher)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        if log:
            self.print_input_output_info()

    @property
    def inputs(self):
        if self.network:
            return self.network.input_info if hasattr(self.network, 'input_info') else self.network.inputs
        return self.exec_network.input_info if hasattr(self.exec_network, 'input_info') else self.exec_network.inputs
