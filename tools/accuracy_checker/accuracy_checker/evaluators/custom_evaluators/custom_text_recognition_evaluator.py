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

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations
from ...launcher import create_launcher
from ...logging import print_info
from ...progress_reporters import ProgressReporter
from ...representation import CharacterRecognitionPrediction, CharacterRecognitionAnnotation


class TextRecognitionWithAttentionEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, model, lowercase):
        self.dataset_config = dataset_config
        self.preprocessing_executor = None
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self.model = model
        self.lowercase = lowercase
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        dataset_config = config['datasets']
        launcher = create_launcher(config['launchers'][0], delayed_model_loading=True)
        lowercase = config.get('lowercase', False)
        model_type = config.get('model_type', 'SequentialFormulaRecognitionModel')
        if model_type not in MODEL_TYPES.keys():
            raise ValueError(f'Model type {model_type} is not supported')
        model = MODEL_TYPES[model_type](
            config.get('network_info', {}),
            launcher,
            config.get('_models', []),
            {},
            config.get('_model_is_blob'),
            delayed_model_loading=delayed_model_loading
        )
        return cls(dataset_config, launcher, model, lowercase)

    def process_dataset(
            self, subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            calculate_metrics=True,
            **kwargs
    ):
        self._prepare_dataset(dataset_tag)
        self._create_subset(subset, num_images, allow_pairwise_subset)
        self._annotations, self._predictions = [], []
        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )
        compute_intermediate_metric_res = kwargs.get('intermediate_metrics_results', False)
        if compute_intermediate_metric_res:
            metric_interval = kwargs.get('metrics_interval', 1000)
            ignore_results_formatting = kwargs.get('ignore_results_formatting', False)
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_data, batch_meta = extract_image_representations(batch_inputs)
            temporal_output_callback = None
            if output_callback:
                temporal_output_callback = partial(output_callback,
                                                   metrics_result=None,
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
            metrics_result = None
            if self.metric_executor and calculate_metrics:
                metrics_result, _ = self.metric_executor.update_metrics_on_batch(
                    batch_input_ids, batch_annotation, batch_prediction
                )
                if self.metric_executor.need_store_predictions:
                    self._annotations.extend(batch_annotation)
                    self._predictions.extend(batch_prediction)

            if output_callback:
                output_callback(
                    batch_raw_prediction,
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids
                )
            if _progress_reporter:
                _progress_reporter.update(batch_id, len(batch_prediction))
                if compute_intermediate_metric_res and _progress_reporter.current % metric_interval == 0:
                    self.compute_metrics(
                        print_results=True, ignore_results_formatting=ignore_results_formatting
                    )

        if _progress_reporter:
            _progress_reporter.finish()

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []

        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting)

        return self._metrics_results

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(False, ignore_results_formatting)

        result_presenters = self.metric_executor.get_metric_presenters()
        extracted_results, extracted_meta = [], []
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            result, metadata = presenter.extract_result(metric_result)
            if isinstance(result, list):
                extracted_results.extend(result)
                extracted_meta.extend(metadata)
            else:
                extracted_results.append(result)
                extracted_meta.append(metadata)
            if print_results:
                presenter.write_result(metric_result, ignore_results_formatting)

        return extracted_results, extracted_meta

    def print_metrics_results(self, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting)
            return
        result_presenters = self.metric_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_result(metric_result, ignore_results_formatting)

    def _prepare_dataset(self, dataset_tag=''):
        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)

        if self.dataset.batch is None:
            self.dataset.batch = 1

    def _create_subset(self, subset=None, num_images=None, allow_pairwise=False):
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise)

    @property
    def dataset_size(self):
        return self.dataset.size

    def release(self):
        self.model.release()
        self.launcher.release()

    def reset(self):
        self.metric_executor.reset()
        self.model.reset()

    @ staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        model_name = config['name']
        dataset_config = module_specific_params['datasets'][0]
        launcher_config = module_specific_params['launchers'][0]
        return (
            model_name, launcher_config['framework'], launcher_config['device'],
            launcher_config.get('tags'),
            dataset_config['name']
        )

    def register_metric(self, metric_config):
        if isinstance(metric_config, str):
            self.metric_executor.register_metric({'type': metric_config})
        elif isinstance(metric_config, dict):
            self.metric_executor.register_metric(metric_config)
        else:
            raise ValueError('Unsupported metric configuration type {}'.format(type(metric_config)))

    def register_postprocessor(self, postprocessing_config):
        pass

    def register_dumped_annotations(self):
        pass

    def select_dataset(self, dataset_tag):
        if self.dataset is not None and isinstance(self.dataset_config, list):
            return
        dataset_attributes = create_dataset_attributes(self.dataset_config, dataset_tag)
        self.dataset, self.metric_executor, self.preprocessor, self.postprocessor = dataset_attributes
        if self.model.vocab is None:
            self.model.vocab = self.dataset.metadata.get('vocab')

    def load_network(self, network=None):
        self.model.load_network(network, self.launcher)

    def load_network_from_ir(self, models_dict):
        self.model.load_model(models_dict, self.launcher)

    def get_network(self):
        return self.model.get_network()

    def get_metrics_attributes(self):
        if not self.metric_executor:
            return {}
        return self.metric_executor.get_metrics_attributes()

    @staticmethod
    def _create_progress_reporter(check_progress, dataset_size):
        pr_kwargs = {}
        if isinstance(check_progress, int) and not isinstance(check_progress, bool):
            pr_kwargs = {"print_interval": check_progress}

        return ProgressReporter.provide('print', dataset_size, **pr_kwargs)


class BaseModel:
    def __init__(self, network_info, launcher, default_model_suffix):
        self.default_model_suffix = default_model_suffix
        self.network_info = network_info

    def predict(self, identifiers, input_data):
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
        if not delayed_model_loading:
            model, weights = self.automatic_model_search(network_info)
            if weights is not None:
                self.network = launcher.read_network(str(model), str(weights))
                self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
            else:
                self.exec_network = launcher.ie_core.import_network(str(model))
            self.print_input_output_info()
        self.inputs_mapping = inputs_mapping
        self.outputs_mapping = outputs_mapping

    def predict(self, inputs, identifiers=None):
        return self.exec_network.infer(inputs)

    def release(self):
        del self.exec_network

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        if log:
            self.print_input_output_info()

    def load_network(self, network, launcher):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)

    def get_network(self):
        return self.network


MODEL_TYPES = {
    'SequentialTextRecognitionModel': SequentialTextRecognitionModel,
    'SequentialFormulaRecognitionModel': SequentialFormulaRecognitionModel,
}
