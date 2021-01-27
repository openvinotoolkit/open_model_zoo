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
import pickle
from functools import partial
from collections import OrderedDict
import numpy as np

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher import create_launcher
from ...utils import contains_all, contains_any, extract_image_representations, read_pickle, get_path
from ...progress_reporters import ProgressReporter
from ...logging import print_info


class AutomaticSpeechRecognitionEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, model):
        self.dataset_config = dataset_config
        self.preprocessing_executor = None
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self.model = model
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'device' not in launcher_config:
            launcher_config['device'] = 'CPU'

        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        model = ASRModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model)

    def process_dataset(
            self, subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            calculate_metrics=True,
            **kwargs):
        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)

        self._annotations, self._predictions = [], []

        self._create_subset(subset, num_images, allow_pairwise_subset)
        metric_config = self.configure_intermediate_metrics_results(kwargs)
        compute_intermediate_metric_res, metric_interval, ignore_results_formatting = metric_config

        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):

            batch_features = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_features)
            encoder_callback = None
            if output_callback:
                encoder_callback = partial(output_callback,
                                           metrics_result=None,
                                           element_identifiers=batch_identifiers,
                                           dataset_indices=batch_input_ids)

            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_inputs_extr, encoder_callback=encoder_callback
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
                    batch_raw_prediction[0],
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

        if self.model.store_encoder_predictions:
            self.model.save_encoder_predictions()

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
        if self.metric_executor:
            self.metric_executor.reset()
        if hasattr(self, '_annotations'):
            del self._annotations
            del self._predictions
            del self._input_ids
        del self._metrics_results
        self._annotations = []
        self._predictions = []
        self._input_ids = []
        self._metrics_results = []
        if self.dataset:
            self.dataset.reset(self.postprocessor.has_processors)

    @staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        model_name = config['name']
        dataset_config = module_specific_params['datasets'][0]
        launcher_config = module_specific_params['launchers'][0]
        return (
            model_name, launcher_config['framework'], launcher_config['device'], launcher_config.get('tags'),
            dataset_config['name']
        )

    def _create_subset(self, subset=None, num_images=None, allow_pairwise=False):
        if self.dataset.batch is None:
            self.dataset.batch = 1
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise)

    @staticmethod
    def configure_intermediate_metrics_results(config):
        compute_intermediate_metric_res = config.get('intermediate_metrics_results', False)
        metric_interval, ignore_results_formatting = None, None
        if compute_intermediate_metric_res:
            metric_interval = config.get('metrics_interval', 1000)
            ignore_results_formatting = config.get('ignore_results_formatting', False)
        return compute_intermediate_metric_res, metric_interval, ignore_results_formatting

    def load_network(self, network=None):
        self.model.load_network(network, self.launcher)

    def load_network_from_ir(self, models_list):
        self.model.load_model(models_list, self.launcher)

    def get_network(self):
        return self.model.get_network()

    def get_metrics_attributes(self):
        if not self.metric_executor:
            return {}
        return self.metric_executor.get_metrics_attributes()

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

    @staticmethod
    def _create_progress_reporter(check_progress, dataset_size):
        pr_kwargs = {}
        if isinstance(check_progress, int) and not isinstance(check_progress, bool):
            pr_kwargs = {"print_interval": check_progress}

        return ProgressReporter.provide('print', dataset_size, **pr_kwargs)


class BaseModel:
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher

    def predict(self, idenitifiers, input_data):
        raise NotImplementedError

    def release(self):
        pass


# pylint: disable=E0203
class BaseDLSDKModel:
    def _reshape_input(self, input_shapes):
        del self.exec_network
        self.network.reshape(input_shapes)
        self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)

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
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        print_info('{} - Found weights: {}'.format(self.default_model_suffix, weights))
        return model, weights

    def load_network(self, network, launcher):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.input_blob is None:
                output_blob = next(iter(self.exec_network.outputs))
            else:
                output_blob = (
                    '_'.join([self.default_model_suffix, self.output_blob])
                    if with_prefix else self.output_blob.split(self.default_model_suffix + '_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix

    def load_model(self, network_info, launcher, log=False):
        if 'onnx_model' in network_info:
            network_info.update(launcher.config)
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        self.set_input_and_output()
        if log:
            self.print_input_output_info()


def create_encoder(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': EncoderDLSDKModel,
        'onnx_runtime': EncoderONNXModel,
        'dummy': DummyEncoder
    }
    framework = launcher.config['framework']
    if 'predictions' in model_config and not model_config.get('store_predictions', False):
        framework = 'dummy'
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


def create_prediction(model_config, launcher, delayed_model_loading):
    launcher_model_mapping = {
        'dlsdk': PredictionDLSDKModel,
        'onnx_runtime': PredictionONNXModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)

def create_joint(model_config, launcher, delayed_model_loading):
    launcher_model_mapping = {
        'dlsdk': JointDLSDKModel,
        'onnx_runtime': JointONNXModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)

class ASRModel(BaseModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        if models_args and not delayed_model_loading:
            encoder = network_info.get('encoder', {})
            prediction = network_info.get('prediction', {})
            joint = network_info.get('joint', {})
            if not contains_any(encoder, ['model', 'onnx_model']) and models_args:
                encoder['model'] = models_args[0]
                encoder['_model_is_blob'] = is_blob
            if not contains_any(prediction, ['model', 'onnx_model']) and models_args:
                prediction['model'] = models_args[1 if len(models_args) > 1 else 0]
                prediction['_model_is_blob'] = is_blob
            if not contains_any(joint, ['model', 'onnx_model']) and models_args:
                joint['model'] = models_args[2 if len(models_args) > 2 else 0]
                joint['_model_is_blob'] = is_blob
            network_info.update({'encoder': encoder, 'prediction': prediction, 'joint': joint})
        if not contains_all(network_info, ['encoder', 'prediction', 'joint']) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder, prediction and joint fields')
        # self.num_processing_frames = network_info['decoder'].get('num_processing_frames', 16)
        # self.processing_frames_buffer = []
        self.encoder = create_encoder(network_info['encoder'], launcher, delayed_model_loading)
        self.prediction = create_prediction(network_info['prediction'], launcher, delayed_model_loading)
        self.joint = create_joint(network_info['joint'], launcher, delayed_model_loading)
        self.store_encoder_predictions = network_info['encoder'].get('store_predictions', False)
        self._encoder_predictions = [] if self.store_encoder_predictions else None
        self._part_by_name = {'encoder': self.encoder, 'prediction': self.decoder, 'joint': self.joint}
        self._raw_outs = OrderedDict()
        self.adapter = create_adapter(network_info.get('adapter', 'dumb_decoder'))

        self._blank_id = 28
        self._SOS = -1
        self._max_symbols_per_step = 30

    def predict(self, identifiers, input_data, encoder_callback=None):
        predictions, raw_outputs = [], []
        for data in input_data:
            encoder_prediction, decoder_inputs = self.encoder.predict(identifiers, data)
            if encoder_callback:
                encoder_callback(encoder_prediction)
            if self.store_encoder_predictions:
                self._encoder_predictions.append(encoder_prediction)
            raw_output, prediction = self.decoder(identifiers, decoder_inputs)
            raw_outputs.append(raw_output)
            predictions.append(prediction)
        return raw_outputs, predictions

    def reset(self):
        self.processing_frames_buffer = []
        if self._encoder_predictions is not None:
            self._encoder_predictions = []

    def release(self):
        self.encoder.release()
        self.prediction.release()
        self.joint.release()

    def save_encoder_predictions(self):
        if self._encoder_predictions is not None:
            prediction_file = Path(self.network_info['encoder'].get('predictions', 'encoder_predictions.pickle'))
            with prediction_file.open('wb') as file:
                pickle.dump(self._encoder_predictions, file)

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, launcher)

    def _add_raw_encoder_predictions(self, encoder_prediction):
        for key, output in encoder_prediction.items():
            if key not in self._raw_outs:
                self._raw_outs[key] = []
            self._raw_outs[key].append(output)

    def get_network(self):
        return [{'name': 'encoder', 'model': self.encoder.network},
                {'name': 'prediction', 'model': self.prediction.network},
                {'name': 'joint', 'model': self.joint.network}]

    def decoder(self, identifiers, logits):
        output = []
        raw_outputs = []
        batches = logits.shape[0]
        for batch_idx in range(batches):
            inseq = np.squeeze(logits[batch_idx, :, :])
            # inseq: TxBxF
            logitlen = inseq.shape[0]
            sentence = self._greedy_decode(inseq, logitlen)
            output.append(sentence)
            # raw_outputs.append(raw_output)
        result = self.adapter.process(output, identifiers, [{}])

        return raw_outputs, result

    def _greedy_decode(self, x, out_len):
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
                hidden_prime = (g['151'], g['152'])
                g = g['153']
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                k = np.argmax(logp)

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        return label

    def _pred_step(self, label, hidden):
        if label == self._SOS:
            label = self._blank_id
        if label > self._blank_id:
            label -= 1
        inputs = {'input.1': [[label,]], '1': hidden[0], '2': hidden[1]}
        return self.prediction.predict(None, inputs)

    def _joint_step(self, enc, pred, log_normalize=False):
        inputs = {'0': enc, '1': pred}
        logits, logits_blob = self.joint.predict(None, inputs)
        logits = logits_blob[:, 0, 0, :]
        if not log_normalize:
            return logits

        # probs = F.log_softmax(logits, dim=len(logits.shape) - 1)
        # return probs

    def _get_last_symb(self, labels) -> int:
        return self._SOS if len(labels) == 0 else labels[-1]


class CommonDLSDKModel(BaseModel, BaseDLSDKModel):
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        self.input_blob, self.output_blob = None, None
        self.with_prefix = None
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def predict(self, identifiers, input_data):
        input_data = self.fit_to_input(input_data)
        results = self.exec_network.infer(input_data)
        return results, results[self.output_blob]

    def release(self):
        del self.exec_network
        del self.launcher

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
            return fitted
        else:
            return self.fit_one_input(self.input_blob, input_data)

    def fit_one_input(self, input_blob, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info[input_blob].input_data
        else:
            input_info = self.exec_network.inputs[input_blob]
        if tuple(input_info.shape) != np.shape(input_data):
            self._reshape_input({input_blob: np.shape(input_data)})

        return {input_blob: np.array(input_data)}

class EncoderDLSDKModel(CommonDLSDKModel):
    default_model_suffix = 'encoder'

class PredictionDLSDKModel(CommonDLSDKModel):
    default_model_suffix = 'prediction'

class JointDLSDKModel(CommonDLSDKModel):
    default_model_suffix = 'joint'

class CommonONNXModel(BaseModel):
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher, *args, **kwargs):
        super().__init__(network_info, launcher)
        model = self.automatic_model_search(network_info)
        self.inference_session = launcher.create_inference_session(str(model))
        self.input_blob = next(iter(self.inference_session.get_inputs()))
        self.output_blob = next(iter(self.inference_session.get_outputs()))

    def predict(self, identifiers, input_data):
        fitted = self.fit_to_input(input_data)
        results = self.inference_session.run((self.output_blob.name, ), fitted)
        return results, results[0]

    def fit_to_input(self, input_data):
        return {self.input_blob.name: input_data[0]}

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

        return model

class EncoderONNXModel(CommonONNXModel):
    default_model_suffix = 'encoder'

    def fit_to_input(self, input_data):
        frames, _, _ = input_data.shape
        return {self.input_blob.name: input_data, '1': np.array([frames], dtype=np.int64)}

class PredictionONNXModel(CommonONNXModel):
    default_model_suffix = 'prediction'

class JointONNXModel(CommonONNXModel):
    default_model_suffix = 'joint'

class DummyEncoder(BaseModel):
    def __init__(self, network_info, launcher):
        super().__init__(network_info, launcher)
        if 'predictions' not in network_info:
            raise ConfigError('predictions_file is not found')
        self._predictions = read_pickle(network_info['predictions'])
        self.iterator = 0

    def predict(self, identifiers, input_data):
        result = self._predictions[self.iterator]
        self.iterator += 1
        return None, result
