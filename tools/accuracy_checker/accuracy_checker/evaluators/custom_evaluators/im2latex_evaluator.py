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
import numpy as np

from ..base_evaluator import BaseEvaluator
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations
from ...launcher import create_launcher
from ...dataset import Dataset
from ...logging import print_info
from ...metrics import MetricsExecutor
from ...preprocessor import PreprocessingExecutor
from ...representation import CharacterRecognitionPrediction


class Im2latexEvaluator(BaseEvaluator):
    def __init__(self, dataset, preprocessing, metric_executor, launcher, model):
        self.dataset = dataset
        self.preprocessing_executor = preprocessing
        self.metric_executor = metric_executor
        self.launcher = launcher
        self.model = model
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config):
        dataset_config = config['datasets'][0]
        dataset = Dataset(dataset_config)
        preprocessing = PreprocessingExecutor(dataset_config.get('preprocessing', []), dataset.name)
        metrics_executor = MetricsExecutor(dataset_config['metrics'], dataset)
        launcher = create_launcher(config['launchers'][0], delayed_model_loading=True)
        meta = dataset.metadata
        model = SequentialModel(
            config.get('network_info', {}),
            launcher,
            config.get('_models', []),
            meta,
            config.get('_model_is_blob'),
        )
        return cls(dataset, preprocessing, metrics_executor, launcher, model)

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        self._annotations, self._predictions = [], []
        compute_intermediate_metric_res = kwargs.get('intermediate_metrics_results', False)
        if compute_intermediate_metric_res:
            metric_interval = kwargs.get('metrics_interval', 1000)
            ignore_results_formatting = kwargs.get('ignore_results_formatting', False)

        if progress_reporter:
            progress_reporter.reset(self.dataset.size)
        self.dataset_meta = self.dataset.metadata
        for batch_id, (_, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):

            batch_inputs = self.preprocessing_executor.process(batch_inputs, batch_annotation)
            batch_inputs, _ = extract_image_representations(batch_inputs)
            batch_prediction = self.model.predict(batch_identifiers, batch_inputs)
            batch_prediction = [CharacterRecognitionPrediction(
                label=batch_prediction, identifier=batch_annotation[0].identifier)]
            self._annotations.extend(batch_annotation)
            self._predictions.extend(batch_prediction)

            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_inputs))
                if compute_intermediate_metric_res and progress_reporter.current % metric_interval == 0:
                    self.compute_metrics(print_results=True, ignore_results_formatting=ignore_results_formatting)

        if progress_reporter:
            progress_reporter.finish()

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

    def release(self):
        self.model.release()
        self.launcher.release()

    def reset(self):
        self.metric_executor.reset()
        self.model.reset()

    @staticmethod
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
            print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        if model.suffix == '.blob':
            return model, None
        weights = network_info.get('weights', model.parent / model.name.replace('xml', 'bin'))
        if 'weights' not in network_info:
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



def create_recognizer(model_config, launcher, suffix):
    """Creates encoder and decoder element for Im2LaTeX model

    Args:
        model_config (dict): paths to xml and bin file
        launcher: launcher object (e.g. DLSDK)
        suffix: encoder or decoder

    Raises:
        ValueError: If wrong framework is passed

    Returns:
        Created model object
    """
    launcher_model_mapping = {
        'dlsdk': RecognizerDLSDKModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, suffix)


class SequentialModel:
    def __init__(self, network_info, launcher, models_args, meta, is_blob=None):
        recognizer_encoder = network_info.get('recognizer_encoder', {})
        recognizer_decoder = network_info.get('recognizer_decoder', {})
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
        self.vocab = meta['vocab']
        self.recognizer_encoder = create_recognizer(network_info['recognizer_encoder'], launcher, 'encoder')
        self.recognizer_decoder = create_recognizer(network_info['recognizer_decoder'], launcher, 'decoder')
        self.sos_index = 0
        self.eos_index = 2
        self.max_seq_len = int(network_info['max_seq_len'])

    def get_phrase(self, indices):
        res = ''
        for idx in indices:
            if idx != self.eos_index:
                res += ' ' + str(self.vocab.get(idx, '?'))
            else:
                return res.strip()
        return res.strip()

    def predict(self, identifiers, input_data):
        assert len(identifiers) == 1
        input_data = np.array(input_data)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        enc_res = self.recognizer_encoder.predict(
            inputs={'imgs': input_data})
        row_enc_out = enc_res['row_enc_out']
        dec_states_h = enc_res['hidden']
        dec_states_c = enc_res['context']
        O_t = enc_res['init_0']

        tgt = np.array([[self.sos_index]])
        logits = []
        for _ in range(self.max_seq_len):

            dec_res = self.recognizer_decoder.predict(inputs={'row_enc_out': row_enc_out,
                                                              'dec_st_c': dec_states_c, 'dec_st_h': dec_states_h,
                                                              'output_prev': O_t, 'tgt': tgt
                                                              })

            dec_states_h = dec_res['dec_st_h_t']
            dec_states_c = dec_res['dec_st_c_t']
            O_t = dec_res['output']
            logit = dec_res['logit']
            logits.append(logit)
            tgt = np.array([[np.argmax(np.array(logit), axis=1)]])

            if tgt[0][0][0] == self.eos_index:
                break

        logits = np.array(logits)
        logits = logits.squeeze(axis=1)
        targets = np.argmax(logits, axis=1)
        result_phrase = self.get_phrase(targets)
        return result_phrase

    def reset(self):
        pass

    def release(self):
        self.recognizer_encoder.release()
        self.recognizer_decoder.release()


class RecognizerDLSDKModel(BaseModel):
    def __init__(self, network_info, launcher, suffix):
        super().__init__(network_info, launcher, suffix)
        model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.print_input_output_info()

    def predict(self, inputs, identifiers=None):
        return self.exec_network.infer(inputs)

    def release(self):
        del self.exec_network
