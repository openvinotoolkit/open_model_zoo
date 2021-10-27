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

from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher import create_launcher
from ...utils import contains_all, contains_any, read_pickle, get_path, extract_image_representations
from ...logging import print_info
from ..base_evaluator import BaseEvaluator
from ...progress_reporters import ProgressReporter
from ..quantization_model_evaluator import create_dataset_attributes


class OpenNMTEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        self.dataset_config = dataset_config
        self.preprocessing_executor = None
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self.model = model
        self._metrics_results = []
        self.config = orig_config

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'device' not in launcher_config:
            launcher_config['device'] = 'CPU'

        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        model = OpenNMTModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

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
        (compute_intermediate_metric_res, metric_interval, ignore_results_formatting,
         ignore_metric_reference) = metric_config

        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_inputs)
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
                        print_results=True, ignore_results_formatting=ignore_results_formatting,
                        ignore_metric_reference=ignore_metric_reference
                    )
                    self.write_results_to_csv(kwargs.get('csv_result'), ignore_results_formatting, metric_interval)

        if _progress_reporter:
            _progress_reporter.finish()

        if self.model.store_encoder_predictions:
            self.model.save_encoder_predictions()

    def compute_metrics(self, print_results=True, ignore_results_formatting=False, ignore_metric_reference=False):
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []

        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting, ignore_metric_reference)

        return self._metrics_results

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False,
                                ignore_metric_reference=False):
        if not self._metrics_results:
            self.compute_metrics(False, ignore_results_formatting, ignore_metric_reference)

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
                presenter.write_result(metric_result, ignore_results_formatting, ignore_metric_reference)

        return extracted_results, extracted_meta

    def print_metrics_results(self, ignore_results_formatting=False, ignore_metric_reference=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting, ignore_metric_reference)
            return
        result_presenters = self.metric_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_result(metric_result, ignore_results_formatting, ignore_metric_reference)

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
        metric_interval, ignore_results_formatting, ignore_metric_reference = None, None, None
        if compute_intermediate_metric_res:
            metric_interval = config.get('metrics_interval', 1000)
            ignore_results_formatting = config.get('ignore_results_formatting', False)
            ignore_metric_reference = config.get('ignore_metric_reference', False)
        return compute_intermediate_metric_res, metric_interval, ignore_results_formatting, ignore_metric_reference

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

    @property
    def dataset_size(self):
        return self.dataset.size

    def send_processing_info(self, sender):
        if not sender:
            return {}
        model_type = None
        details = {}
        metrics = self.dataset_config[0].get('metrics', [])
        metric_info = [metric['type'] for metric in metrics]
        adapter_type = self.model.adapter.__provider__
        details.update({
            'metrics': metric_info,
            'model_file_type': model_type,
            'adapter': adapter_type,
        })
        if self.dataset is None:
            self.select_dataset('')

        details.update(self.dataset.send_annotation_info(self.dataset_config[0]))
        return details


# class ASREvaluator(AutomaticSpeechRecognitionEvaluator):
#     @classmethod
#     def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
#         dataset_config = config['datasets']
#         launcher_config = config['launchers'][0]
#         if launcher_config['framework'] == 'dlsdk' and 'device' not in launcher_config:
#             launcher_config['device'] = 'CPU'
#
#         launcher = create_launcher(launcher_config, delayed_model_loading=True)
#         model = ASRModel(
#             config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
#             delayed_model_loading
#         )
#         return cls(dataset_config, launcher, model, orig_config)
#

class BaseModel:
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher

    def predict(self, identifiers, input_data):
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

    def load_network(self, network, launcher):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)

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

def create_model(model_name, model_config, launcher, delayed_model_loading=False):
    model_mapping = {
        'encoder': {
            'dlsdk': EncoderDLSDKModel,
            'onnx_runtime': EncoderONNXModel,
            'dummy': DummyModel
        },
        'decoder': {
            'dlsdk': DecoderDLSDKModel,
            'onnx_runtime': DecoderONNXModel,
            'dummy': DummyModel
        },

        'generator':  {
            'dlsdk': GeneratorDLSDKModel,
            'onnx_runtime': GeneratorONNXModel
        }
    }
    framework = launcher.config['framework']
    if 'predictions' in model_config and not model_config.get('store_predictions', False):
        framework = 'dummy'
    launcher_model_mapping = model_mapping.get(model_name)
    if not launcher_model_mapping:
        raise ValueError('model {} is not supported'.format(model_name))
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model {} for framework {} is not supported'.format(model_name, framework))
    return model_class(model_config, launcher, delayed_model_loading)


class OpenNMTModel(BaseModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        if models_args and not delayed_model_loading:
            encoder = network_info.get('encoder', {})
            decoder = network_info.get('decoder', {})
            generator = network_info.get('generator', {})
            if not contains_any(encoder, ['model', 'onnx_model']) and models_args:
                encoder['model'] = models_args[0]
                encoder['_model_is_blob'] = is_blob
            if not contains_any(decoder, ['model', 'onnx_model']) and models_args:
                decoder['model'] = models_args[1 if len(models_args) > 1 else 0]
                decoder['_model_is_blob'] = is_blob
            if not contains_any(generator, ['model', 'onnx_model']) and models_args:
                generator['model'] = models_args[2 if len(models_args) > 2 else 0]
                generator['_model_is_blob'] = is_blob
            network_info.update({'encoder': encoder, 'decoder': decoder, 'generator': generator})
        if not contains_all(network_info, ['encoder', 'decoder', 'generator']) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder, decoder and generator fields')
        self.encoder = create_model('encoder', network_info['encoder'], launcher, delayed_model_loading)
        self.decoder = create_model('decoder', network_info['decoder'], launcher, delayed_model_loading)
        self.generator = create_model('generator', network_info['generator'], launcher, delayed_model_loading)
        self.store_encoder_predictions = network_info['encoder'].get('store_predictions', False)
        self.store_decoder_predictions = network_info['decoder'].get('store_predictions', False)
        self._encoder_predictions = [] if self.store_encoder_predictions else None
        self._decoder_predictions = [] if self.store_decoder_predictions else None
        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder, 'generator': self.generator}
        self._raw_outs = OrderedDict()
        self.adapter = create_adapter(network_info.get('adapter', 'dumb_decoder'))
        # self.adapter = None

        self.beams = 5
        self.pad = 1
        self.bos = 2
        self.eos = 3
        self.unk = 0
        self.nbest = 1
        self.max_length = 100
        self.min_length = 0
        self.batch = 1

        self.alpha = 0
        self.beta = 0
        self.length_penalty = None
        self.coverage_penalty = None
        self.global_scorer = GNMTGlobalScorer(self.alpha, self.beta, self.length_penalty, self.coverage_penalty)
        self.src_vocabs = []

    def predict(self, identifiers, input_data, encoder_callback=None):
        predictions, raw_outputs = [], []

        decode_strategy = BeamSearch(
            self.beams,
            batch_size=self.batch,
            pad=self.pad,
            bos=self.bos,
            eos=self.eos,
            unk=self.unk,
            n_best=self.nbest,
            global_scorer=self.global_scorer,
            min_length=self.min_length,
            max_length=self.max_length,
            return_attention=False,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            stepwise_penalty=False,
            ratio=0.0,
            ban_unk_token=False,
        )


        for data in input_data:
            encoder_state, memory, src_len = self.encoder.predict(identifiers, data)
            if encoder_callback:
                encoder_callback((encoder_state, memory, src_len))
            if self.store_encoder_predictions:
                self._encoder_predictions.append((encoder_state, memory, src_len))
            src_map = None
            (
                fn_map_state,
                memory_bank,
                memory_lengths,
                src_map,
            ) = decode_strategy.initialize(memory, src_len, src_map)
            self.decoder.init_state(encoder_state)
            if fn_map_state is not None:
                self.decoder.map_state(fn_map_state)

            for step in range(decode_strategy.max_length):
                # decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
                decoder_input = decode_strategy.current_predictions.view().reshape([1, self.beams, 1])
                # decoder_input = decode_strategy.get_decoder_input()

                log_probs, attn = self._decode_and_generate(
                    decoder_input,
                    memory_bank,
                    self.batch,
                    self.src_vocabs,
                    memory_lengths=memory_lengths,
                    src_map=src_map,
                    step=step,
                    batch_offset=decode_strategy.batch_offset,
                )

                decode_strategy.advance(log_probs, attn)
                any_finished = decode_strategy.is_finished.any()
                if any_finished:
                    decode_strategy.update_finished()
                    if decode_strategy.done:
                        break

                select_indices = decode_strategy.select_indices.squeeze()

                if any_finished:
                    # Reorder states.
                    if isinstance(memory_bank, tuple):
                        # memory_bank = tuple(
                        #     x.index_select(1, select_indices) for x in memory_bank
                        # )
                        memory_bank = tuple(
                            np.take(x, select_indices, axis=1) for x in memory_bank
                        )
                    else:
                        # memory_bank = memory_bank.index_select(1, select_indices)
                        memory_bank = np.take(memory_bank, select_indices, axis=1)

                    # memory_lengths = memory_lengths.index_select(0, select_indices)
                    memory_lengths = np.take(memory_lengths, select_indices, axis=0)

                    # if src_map is not None:
                    #     src_map = src_map.index_select(1, select_indices)

                if self.beams > 1 or any_finished:
                    self.decoder.map_state(
                        lambda state, dim: np.take(state, select_indices, axis=dim)
                    )

                ddd = {
                    'encoder_state': encoder_state,
                    'memory': memory,
                    'src_len': src_len,
                    'step': step,
                    'decoder_input': decoder_input,
                    'decoder_state_h0': self.decoder.state['hidden'][0],
                    'decoder_state_h1': self.decoder.state['hidden'][1],
                    'decoder_state_input_feed': self.decoder.state['input_feed'],
                    'any_finished': any_finished,
                    'select_indices': select_indices,
                    'log_probs': log_probs,
                    'memory_bank': memory_bank,
                }
                np.savez("./debug_data/open_nmt_step_%d.npz" % step, **ddd)
            predictions.append(decode_strategy.predictions)

            # raw_output, prediction = self.decoder(identifiers, decoder_inputs, callback=encoder_callback)
            # raw_outputs.append(raw_output)
            # predictions.append(prediction)
        return raw_outputs, predictions

# self.alive_seq:
    # tensor([[2, 40, 10, 54, 40],
    #         [2, 40, 10, 4, 17],
    #         [2, 40, 10, 54, 39],
    #         [2, 10, 45, 17, 11],
    #         [2, 10, 13, 17, 8]])

# self.select_indices
#     tensor([0, 1, 2, 4, 3])

    # tensor([[2, 40, 10, 54, 40],
    #         [2, 40, 10, 4, 17],
    #         [2, 40, 10, 54, 39],
    #         [2, 10, 13, 17, 8],
    #         [2, 10, 45, 17, 11]])

# self.alive_seq.index_select(0, self.select_indices)

    def _decode_and_generate(
        self,
        decoder_in,
        memory_bank,
        batch,
        src_vocabs,
        memory_lengths,
        src_map=None,
        step=None,
        batch_offset=None,
    ):
        # if self.copy_attn:
        #     # Turn any copied words into UNKs.
        #     decoder_in = decoder_in.masked_fill(
        #         decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
        #     )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch

        # input_feed = self.model.decoder.state["input_feed"].squeeze(0)
        # input_feed_batch, _ = input_feed.size()
        # _, tgt_batch, _ = decoder_in.size()
        # # aeq(tgt_batch, input_feed_batch)
        # # END Additional args check.
        #
        # dec_state = self.model.decoder.state["hidden"]
        # coverage = self.model.decoder.state["coverage"].squeeze(0) \
        #     if self.model.decoder.state["coverage"] is not None else None
        #
        # # dec_out, dec_attn, dec_hidden, dec_input_feed, dec_coverage = self.model.decoder(
        # #     decoder_in, memory_bank, memory_lengths=memory_lengths, step=step,
        # #     input_feed=input_feed, hidden=dec_state, coverage=coverage
        # # )
        # dec_out, dec_attn, dec_hidden, dec_input_feed = self.model.decoder(
        #     decoder_in, memory_bank, memory_lengths=memory_lengths, step=step,
        #     input_feed=input_feed, hidden=dec_state
        # )
        #
        # self.model.decoder.state["hidden"] = dec_hidden
        # self.model.decoder.state["input_feed"] = dec_input_feed
        # # self.model.decoder.state["coverage"] = dec_coverage
        #
        # # IMPORTANT^^^^^^
        # # hidden_in = (h0, c0)
        # # hidden_out = (h1, c1)

        attn, output = self.decoder.predict(decoder_in, memory_bank, memory_lengths)

        # Generator forward.
        # if not self.copy_attn:
        #     if "std" in dec_attn:
        #         attn = dec_attn["std"]
        #     else:
        #         attn = None
        #     log_probs = self.model.generator(dec_out.squeeze(0))
        #
        #     # returns [(batch_size x beam_size) , vocab ] when 1 step
        #     # or [ tgt_len, batch_size, vocab ] when full sentence
        # else:
        #     # attn = dec_attn["copy"]
        #     # scores = self.model.generator(
        #     #     dec_out.view(-1, dec_out.size(2)),
        #     #     attn.view(-1, attn.size(2)),
        #     #     src_map,
        #     # )
        #     # # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
        #     # if batch_offset is None:
        #     #     scores = scores.view(-1, batch.batch_size, scores.size(-1))
        #     #     scores = scores.transpose(0, 1).contiguous()
        #     # else:
        #     #     scores = scores.view(-1, self.beam_size, scores.size(-1))
        #     # scores = collapse_copy_scores(
        #     #     scores,
        #     #     batch,
        #     #     self._tgt_vocab,
        #     #     src_vocabs,
        #     #     batch_dim=0,
        #     #     batch_offset=batch_offset,
        #     # )
        #     # scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
        #     # log_probs = scores.squeeze(0).log()
        #     # # returns [(batch_size x beam_size) , vocab ] when 1 step
        #     # # or [ tgt_len, batch_size, vocab ] when full sentence
        #     pass

        log_probs = self.generator.predict(output.squeeze())

        return log_probs, attn

    def reset(self):
        self.processing_frames_buffer = []
        if self._encoder_predictions is not None:
            self._encoder_predictions = []

    def release(self):
        self.encoder.release()
        self.decoder.release()
        self.generator.release()

    def save_encoder_predictions(self):
        if self._encoder_predictions is not None:
            prediction_file = Path(self.network_info['encoder'].get('predictions', 'encoder_predictions.pickle'))
            with prediction_file.open('wb') as file:
                pickle.dump(self._encoder_predictions, file)

    def save_decoder_predictions(self):
        if self._decoder_predictions is not None:
            prediction_file = Path(self.network_info['decoder'].get('predictions', 'decoder_predictions.pickle'))
            with prediction_file.open('wb') as file:
                pickle.dump(self._encoder_predictions, file)

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, launcher)

    def get_network(self):
        return [{'name': 'encoder', 'model': self.encoder.network},
                {'name': 'decoder', 'model': self.decoder.network},
                {'name': 'generator', 'model': self.generator.network}]

class CommonDLSDKModel(BaseModel, BaseDLSDKModel):
    default_model_suffix = 'encoder'
    input_layers = []
    output_layers = []

    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        self.with_prefix = None
        if not hasattr(self, 'output_blob'):
            self.output_blob = None
        self.input_blob = None
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def predict(self, identifiers, input_data, callback=None):
        input_data = self.fit_to_input(input_data)
        results = self.exec_network.infer(input_data)
        return (results['state.0'], results['state.1']), results['memory'], results['src_len']

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
        else:
            fitted = self.fit_one_input(self.input_blob, input_data)
        return fitted

    def fit_one_input(self, input_blob, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info[input_blob].input_data
        else:
            input_info = self.exec_network.inputs[input_blob]
        if tuple(input_info.shape) != np.shape(input_data):
            self._reshape_input({input_blob: np.shape(input_data)})

        return {input_blob: np.array(input_data)}

# class CommonEncoderModel(CommonDLSDKModel):
#     def fit_to_input(self, input_data):
#         if isinstance(input_data, list):
#             src = np.array(input_data)
#             src_len = np.array([len(input_data),])
#             for _ in range(2):
#                 src = np.expand_dims(src, -1)
#             input_data = {'src': src, 'src_len': src_len}
#         super().fit_to_input(input_data)
#

# mb = np.tile(memory_bank, (1, 5, 1))
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    # perm = list(range(len(x.size())))
    # if dim != 0:
    #     perm[0], perm[dim] = perm[dim], perm[0]
    #     x = x.permute(perm).contiguous()
    # out_size = list(x.size())
    # out_size[0] *= count
    # batch = x.size(0)
    # x = x.view(batch, -1) \
    #      .transpose(0, 1) \
    #      .repeat(count, 1) \
    #      .transpose(0, 1) \
    #      .contiguous() \
    #      .view(*out_size)
    # if dim != 0:
    #     x = x.permute(perm).contiguous()
    # return x
    dims = [1,] * len(x.shape)
    dims[dim] = count
    return np.tile(x, dims)

class DecodeStrategy(object):
    """Base class for generation strategies.

    Args:
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        unk (int): Magic integer in output vocab.
        batch_size (int): Current batch size.
        parallel_paths (int): Decoding strategies like beam search
            use parallel paths. Each batch is repeated ``parallel_paths``
            times in relevant state tensors.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        ban_unk_token (Boolean): Whether unk token is forbidden
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            tokens, it may repeat.
        return_attention (bool): Whether to work with attention too. If this
            is true, it is assumed that the decoder is attentional.

    Attributes:
        pad (int): See above.
        bos (int): See above.
        eos (int): See above.
        unk (int): See above.
        predictions (list[list[LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
            list of scores.
        attention (list[list[FloatTensor or list[]]]): For each
            batch, holds a list of attention sequence tensors
            (or empty lists) having shape ``(step, inp_seq_len)`` where
            ``inp_seq_len`` is the length of the sample (not the max
            length of all inp seqs).
        alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
            This sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        is_finished (ByteTensor or NoneType): Shape
            ``(B, parallel_paths)``. Initialized to ``None``.
        alive_attn (FloatTensor or NoneType): If tensor, shape is
            ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        target_prefix (LongTensor or NoneType): If tensor, shape is
            ``(B x parallel_paths, prefix_seq_len)``, where ``prefix_seq_len``
            is the (max) length of the pre-fixed prediction.
        min_length (int): See above.
        max_length (int): See above.
        ban_unk_token (Boolean): See above.
        block_ngram_repeat (int): See above.
        exclusion_tokens (set[int]): See above.
        return_attention (bool): See above.
        done (bool): See above.
    """

    def __init__(self, pad, bos, eos, unk, batch_size, parallel_paths,
                 global_scorer, min_length, block_ngram_repeat,
                 exclusion_tokens, return_attention, max_length,
                 ban_unk_token):

        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.unk = unk

        self.batch_size = batch_size
        self.parallel_paths = parallel_paths
        self.global_scorer = global_scorer

        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]
        self.hypotheses = [[] for _ in range(batch_size)]

        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length
        self.ban_unk_token = ban_unk_token

        self.block_ngram_repeat = block_ngram_repeat
        n_paths = batch_size * parallel_paths
        self.forbidden_tokens = [dict() for _ in range(n_paths)]

        self.exclusion_tokens = exclusion_tokens
        self.return_attention = return_attention

        self.done = False

    # def get_device_from_memory_bank(self, memory_bank):
    #     if isinstance(memory_bank, tuple):
    #         mb_device = memory_bank[0].device
    #     else:
    #         mb_device = memory_bank.device
    #     return mb_device

    def initialize_tile(self, memory_bank, src_lengths, src_map=None,
                        target_prefix=None):
        def fn_map_state(state, dim):
            return tile(state, self.beam_size, dim=dim)

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, self.beam_size, dim=1)
                                for x in memory_bank)
        elif memory_bank is not None:
            memory_bank = tile(memory_bank, self.beam_size, dim=1)
        if src_map is not None:
            src_map = tile(src_map, self.beam_size, dim=1)

        self.memory_lengths = tile(src_lengths, self.beam_size)
        if target_prefix is not None:
            target_prefix = tile(target_prefix, self.beam_size, dim=1)

        return fn_map_state, memory_bank, src_map, target_prefix

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None,
                   target_prefix=None):
        """DecodeStrategy subclasses should override :func:`initialize()`.

        `initialize` should be called before all actions.
        used to prepare necessary ingredients for decode.
        """
        # if device is None:
        #     device = torch.device('cpu')
        # self.alive_seq = torch.full(
        #     [self.batch_size * self.parallel_paths, 1], self.bos,
        #     dtype=torch.long, device=device)
        self.alive_seq = np.full([self.batch_size * self.parallel_paths, 1], self.bos, dtype=np.long)

        # self.is_finished = torch.zeros(
        #     [self.batch_size, self.parallel_paths],
        #     dtype=torch.uint8, device=device)
        self.is_finished = np.zeros((self.batch_size, self.parallel_paths), dtype=np.uint8)
        # if target_prefix is not None:
        #     seq_len, batch_size, n_feats = target_prefix.size()
        #     assert batch_size == self.batch_size * self.parallel_paths,\
        #         "forced target_prefix should've extend to same number of path!"
        #     target_prefix_words = target_prefix[:, :, 0].transpose(0, 1)
        #     target_prefix = target_prefix_words[:, 1:]  # remove bos
        #
        #     # fix length constraint and remove eos from count
        #     prefix_non_pad = target_prefix.ne(self.pad).sum(dim=-1).tolist()
        #     self.max_length += max(prefix_non_pad)-1
        #     self.min_length += min(prefix_non_pad)-1
        #
        self.target_prefix = target_prefix  # NOTE: forced prefix words
        return None, memory_bank, src_lengths, src_map

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_unk_removed(self, log_probs):
        if self.ban_unk_token:
            log_probs[:, self.unk] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def block_ngram_repeats(self, log_probs):
        """
        We prevent the beam from going in any direction that would repeat any
        ngram of size <block_ngram_repeat> more thant once.

        The way we do it: we maintain a list of all ngrams of size
        <block_ngram_repeat> that is updated each time the beam advances, and
        manually put any token that would lead to a repeated ngram to 0.

        This improves on the previous version's complexity:
           - previous version's complexity: batch_size * beam_size * len(self)
           - current version's complexity: batch_size * beam_size

        This improves on the previous version's accuracy;
           - Previous version blocks the whole beam, whereas here we only
            block specific tokens.
           - Before the translation would fail when all beams contained
            repeated ngrams. This is sure to never happen here.
        """

        # we don't block nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # # we can't block nothing beam's too short
        # if len(self) < self.block_ngram_repeat:
        #     return
        #
        # n = self.block_ngram_repeat - 1
        # for path_idx in range(self.alive_seq.shape[0]):
        #     # we check paths one by one
        #
        #     current_ngram = tuple(self.alive_seq[path_idx, -n:].tolist())
        #     forbidden_tokens = self.forbidden_tokens[path_idx].get(
        #         current_ngram, None)
        #     if forbidden_tokens is not None:
        #         log_probs[path_idx, list(forbidden_tokens)] = -10e20

    def maybe_update_forbidden_tokens(self):
        """We complete and reorder the list of forbidden_tokens"""

        # we don't forbid nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # # we can't forbid nothing if beam's too short
        # if len(self) < self.block_ngram_repeat:
        #     return
        #
        # n = self.block_ngram_repeat
        #
        # forbidden_tokens = list()
        # for path_idx, seq in zip(self.select_indices, self.alive_seq):
        #
        #     # Reordering forbidden_tokens following beam selection
        #     # We rebuild a dict to ensure we get the value and not the pointer
        #     forbidden_tokens.append(
        #         deepcopy(self.forbidden_tokens[path_idx]))
        #
        #     # Grabing the newly selected tokens and associated ngram
        #     current_ngram = tuple(seq[-n:].tolist())
        #
        #     # skip the blocking if any token in current_ngram is excluded
        #     if set(current_ngram) & self.exclusion_tokens:
        #         continue
        #
        #     forbidden_tokens[-1].setdefault(current_ngram[:-1], set())
        #     forbidden_tokens[-1][current_ngram[:-1]].add(current_ngram[-1])
        #
        # self.forbidden_tokens = forbidden_tokens

    def target_prefixing(self, log_probs):
        """Fix the first part of predictions with `self.target_prefix`.

        Args:
            log_probs (FloatTensor): logits of size ``(B, vocab_size)``.

        Returns:
            log_probs (FloatTensor): modified logits in ``(B, vocab_size)``.
        """
        _B, vocab_size = log_probs.shape
        step = len(self)
        if (self.target_prefix is not None and
                step <= self.target_prefix.size(1)):
            # pick_idx = self.target_prefix[:, step - 1].tolist()  # (B)
            # pick_coo = [[path_i, pick] for path_i, pick in enumerate(pick_idx)
            #             if pick not in [self.eos, self.pad]]
            # mask_pathid = [path_i for path_i, pick in enumerate(pick_idx)
            #                if pick in [self.eos, self.pad]]
            # if len(pick_coo) > 0:
            #     pick_coo = torch.tensor(pick_coo).to(self.target_prefix)
            #     pick_fill_value = torch.ones(
            #         [pick_coo.size(0)], dtype=log_probs.dtype)
            #     # pickups: Tensor where specified index were set to 1, others 0
            #     pickups = torch.sparse_coo_tensor(
            #         pick_coo.t(), pick_fill_value,
            #         size=log_probs.size(), device=log_probs.device).to_dense()
            #     # dropdowns: opposite of pickups, 1 for those shouldn't pick
            #     dropdowns = torch.ones_like(pickups) - pickups
            #     if len(mask_pathid) > 0:
            #         path_mask = torch.zeros(_B).to(self.target_prefix)
            #         path_mask[mask_pathid] = 1
            #         path_mask = path_mask.unsqueeze(1).to(dtype=bool)
            #         dropdowns = dropdowns.masked_fill(path_mask, 0)
            #     # Minus dropdowns to log_probs making probabilities of
            #     # unspecified index close to 0
            #     log_probs -= 10000*dropdowns
            pass
        return log_probs

    def maybe_update_target_prefix(self, select_index):
        """We update / reorder `target_prefix` for alive path."""
        if self.target_prefix is None:
            return
        # # prediction step have surpass length of given target_prefix,
        # # no need to further change this attr
        # if len(self) > self.target_prefix.size(1):
        #     return
        # self.target_prefix = self.target_prefix.index_select(0, select_index)

    def advance(self, log_probs, attn):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()


class BeamSearchBase(DecodeStrategy):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B, beam_size,)``. These
            are the scores used for the topk operation.
        memory_lengths (LongTensor): Lengths of encodings. Used for
            masking attentions.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """
    def __init__(self, beam_size, batch_size, pad, bos, eos, unk, n_best,
                 global_scorer, min_length, max_length, return_attention,
                 block_ngram_repeat, exclusion_tokens, stepwise_penalty,
                 ratio, ban_unk_token):
        super(BeamSearchBase, self).__init__(
            pad, bos, eos, unk, batch_size, beam_size, global_scorer,
            min_length, block_ngram_repeat, exclusion_tokens,
            return_attention, max_length, ban_unk_token)
        # beam parameters
        self.beam_size = beam_size
        self.n_best = n_best
        self.ratio = ratio

        # beam state
        # self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self.top_beam_finished = np.zeros([batch_size], dtype=np.uint8)

        # # BoolTensor was introduced in pytorch 1.2
        # try:
        #     self.top_beam_finished = self.top_beam_finished.bool()
        # except AttributeError:
        #     pass
        # self._batch_offset = torch.arange(batch_size, dtype=torch.long)
        self._batch_offset = np.arange(batch_size, dtype=np.long)

        self.select_indices = None
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        self._stepwise_cov_pen = (
            stepwise_penalty and self.global_scorer.has_cov_pen)
        self._vanilla_cov_pen = (
            not stepwise_penalty and self.global_scorer.has_cov_pen)
        self._cov_pen = self.global_scorer.has_cov_pen
        # self._stepwise_cov_pen = False
        # self._vanilla_cov_pen = False
        # self._cov_pen = False

        self.memory_lengths = None

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def initialize_(self, memory_bank, memory_lengths, src_map, device,
                    target_prefix):
        super(BeamSearchBase, self).initialize(
            memory_bank, memory_lengths, src_map, device, target_prefix)

        # self.best_scores = torch.full(
        #     [self.batch_size], -1e10, dtype=torch.float, device=device)
        self.best_scores = np.full([self.batch_size], -1e10, dtype=np.float)
        # self._beam_offset = torch.arange(
        #     0, self.batch_size * self.beam_size, step=self.beam_size,
        #     dtype=torch.long, device=device)
        self._beam_offset = np.arange(0, self.batch_size * self.beam_size, step=self.beam_size, dtype=np.long)
        # self.topk_log_probs = torch.tensor(
        #     [0.0] + [float("-inf")] * (self.beam_size - 1), device=device
        # ).repeat(self.batch_size).reshape(self.batch_size, self.beam_size)
        self.topk_log_probs = np.asarray([0.0] + [float("-inf")] * (self.beam_size - 1))
        self.topk_log_probs = self.topk_log_probs.repeat(self.batch_size).reshape(self.batch_size, self.beam_size)
        # buffers for the topk scores and 'backpointer'
        # self.topk_scores = torch.empty((self.batch_size, self.beam_size),
        #                                dtype=torch.float, device=device)
        self.topk_scores = np.empty((self.batch_size, self.beam_size), dtype=np.float)
        # self.topk_ids = torch.empty((self.batch_size, self.beam_size),
        #                             dtype=torch.long, device=device)
        self.topk_ids = np.empty((self.batch_size, self.beam_size), dtype=np.long)
        # self._batch_index = torch.empty([self.batch_size, self.beam_size],
        #                                 dtype=torch.long, device=device)
        self._batch_index = np.empty([self.batch_size, self.beam_size], dtype=np.long)

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    @property
    def batch_offset(self):
        return self._batch_offset

    def _pick(self, log_probs, out=None):
        """Take a token pick decision for a step.

        Args:
            log_probs (FloatTensor): (B * beam_size, vocab_size)
            out (Tensor, LongTensor): output buffers to reuse, optional.

        Returns:
            topk_scores (FloatTensor): (B, beam_size)
            topk_ids (LongTensor): (B, beam_size)
        """
        # vocab_size = log_probs.size(-1)
        vocab_size = log_probs.shape[-1]
        # maybe fix some prediction at this step by modifying log_probs
        log_probs = self.target_prefixing(log_probs)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)
        # if out is not None:
        #     torch.topk(curr_scores, self.beam_size, dim=-1, out=out)
        #     return
        # topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1)
        topk_ids = np.argsort(curr_scores)[..., range(self.beam_size * vocab_size - 1, self.beam_size * (vocab_size - 1) - 1, -1)]
        topk_scores = curr_scores[..., topk_ids.squeeze()]
        return topk_scores, topk_ids

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        # self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        self.topk_log_probs[self.is_finished] = -1e10
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        # self.is_finished = self.is_finished.to('cpu')
        # self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        self.top_beam_finished |= self.is_finished[:, 0]
        # # decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
        # decoder_input = decode_strategy.current_predictions.view().reshape([1, self.beams, 1])
        # predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        predictions = self.alive_seq.view().reshape(_B_old, self.beam_size, step)
        # attention = (
        #     self.alive_attn.view(
        #         step - 1, _B_old, self.beam_size, self.alive_attn.size(-1))
        #     if self.alive_attn is not None else None)
        attention = None
        non_finished_batch = []
        # for i in range(self.is_finished.size(0)):  # Batch level
        for i in range(self.is_finished.shape[0]):  # Batch level
            b = self._batch_offset[i]
            # finished_hyp = self.is_finished[i].nonzero(as_tuple=False).view(-1)
            finished_hyp = self.is_finished[i].nonzero()[0]
            # Store finished hypotheses for this batch.
            for j in finished_hyp:  # Beam level: finished beam j in batch i
                if self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start_token.
                    attention[:, i, j, :self.memory_lengths[i]]
                    if attention is not None else None))
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self.memory_lengths[i] * self.ratio
                finish_flag = ((self.topk_scores[i, 0] / pred_len)
                               <= self.best_scores[b]) or \
                    self.is_finished[i].all()
            else:
                finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)  # ``(batch, n_best,)``
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        # non_finished = torch.tensor(non_finished_batch)
        non_finished = np.asarray(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        self.remove_finished_batches(_B_new, _B_old, non_finished,
                                     predictions, attention, step)

    def remove_finished_batches(self, _B_new, _B_old, non_finished,
                                predictions, attention, step):
        # Remove finished batches for the next step.

        # self.top_beam_finished = self.top_beam_finished.index_select(
        #     0, non_finished)
        self.top_beam_finished = np.take(self.top_beam_finished, non_finished, axis=0)

        # self._batch_offset = self._batch_offset.index_select(0, non_finished)
        self._batch_offset = np.take(self._batch_offset, non_finished, axis=0)

        # non_finished = non_finished.to(self.topk_ids.device)

        # self.topk_log_probs = self.topk_log_probs.index_select(0,
        #                                                        non_finished)
        self.topk_log_probs = np.take(self.topk_log_probs, non_finished, axis=0)

        # self._batch_index = self._batch_index.index_select(0, non_finished)
        self._batch_index = np.take(self._batch_index, non_finished, axis=0)

        # self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.select_indices = self._batch_index.view().reshape([_B_new * self.beam_size,])

        # self.alive_seq = predictions.index_select(0, non_finished) \
        #     .view(-1, self.alive_seq.size(-1))
        self.alive_seq = np.take(predictions, non_finished, axis=0).view().reshape([-1, self.alive_seq.shape[-1]])

        # self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_scores = np.take(self.topk_scores, non_finished, axis=0)

        # self.topk_ids = self.topk_ids.index_select(0, non_finished)
        self.topk_ids = np.take(self.topk_ids, non_finished, axis=0)

        self.maybe_update_target_prefix(self.select_indices)
        # if self.alive_attn is not None:
        #     inp_seq_len = self.alive_attn.size(-1)
        #     self.alive_attn = attention.index_select(1, non_finished) \
        #         .view(step - 1, _B_new * self.beam_size, inp_seq_len)
        #     if self._cov_pen:
        #         self._coverage = self._coverage \
        #             .view(1, _B_old, self.beam_size, inp_seq_len) \
        #             .index_select(1, non_finished) \
        #             .view(1, _B_new * self.beam_size, inp_seq_len)
        #         if self._stepwise_cov_pen:
        #             self._prev_penalty = self._prev_penalty.index_select(
        #                 0, non_finished)

    def advance(self, log_probs, attn):
        vocab_size = log_probs.shape[-1]

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        # if self._stepwise_cov_pen and self._prev_penalty is not None:
        #     self.topk_log_probs += self._prev_penalty
        #     self.topk_log_probs -= self.global_scorer.cov_penalty(
        #         self._coverage + attn, self.global_scorer.beta).view(
        #         _B, self.beam_size)

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)
        self.ensure_unk_removed(log_probs)

        # Multiply probs by the beam probability.
        # log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)
        log_probs += self.topk_log_probs.view().reshape((_B * self.beam_size, 1))
        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(
            step + 1, alpha=self.global_scorer.alpha)

        curr_scores = log_probs / length_penalty

        # Avoid any direction that would repeat unwanted ngrams
        self.block_ngram_repeats(curr_scores)

        # Pick up candidate token by curr_scores
        self.topk_scores, self.topk_ids = self._pick(curr_scores)

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        # torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)
        self.topk_log_probs = self.topk_scores * length_penalty

        # Resolve beam origin and map to batch index flat representation.
        self._batch_index = self.topk_ids // vocab_size
        # self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self._batch_index += np.expand_dims(self._beam_offset[:_B], axis=1)
        self.select_indices = self._batch_index.view().reshape((_B * self.beam_size, 1))
        # self.topk_ids.fmod_(vocab_size)  # resolve true word ids
        self.topk_ids = np.fmod(self.topk_ids, vocab_size)  # resolve true word ids

        # Append last prediction.
        # self.alive_seq = torch.cat(
        #     [self.alive_seq.index_select(0, self.select_indices),
        #      self.topk_ids.view(_B * self.beam_size, 1)], -1)
        self.alive_seq = np.concatenate(
            [np.take(self.alive_seq, self.select_indices.squeeze(), 0),
             self.topk_ids.view().reshape((_B * self.beam_size, 1))], axis=-1)

        self.maybe_update_forbidden_tokens()

        # if self.return_attention or self._cov_pen:
        #     current_attn = attn.index_select(1, self.select_indices)
        #     if step == 1:
        #         self.alive_attn = current_attn
        #         # update global state (step == 1)
        #         if self._cov_pen:  # coverage penalty
        #             self._prev_penalty = torch.zeros_like(self.topk_log_probs)
        #             self._coverage = current_attn
        #     else:
        #         self.alive_attn = self.alive_attn.index_select(
        #             1, self.select_indices)
        #         self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)
        #         # update global state (step > 1)
        #         if self._cov_pen:
        #             self._coverage = self._coverage.index_select(
        #                 1, self.select_indices)
        #             self._coverage += current_attn
        #             self._prev_penalty = self.global_scorer.cov_penalty(
        #                 self._coverage, beta=self.global_scorer.beta).view(
        #                     _B, self.beam_size)

        # if self._vanilla_cov_pen:
        #     # shape: (batch_size x beam_size, 1)
        #     cov_penalty = self.global_scorer.cov_penalty(
        #         self._coverage,
        #         beta=self.global_scorer.beta)
        #     self.topk_scores -= cov_penalty.view(_B, self.beam_size).float()

        self.is_finished = np.equal(self.topk_ids, self.eos)
        self.ensure_max_length()


class BeamSearch(BeamSearchBase):
    """
        Beam search for seq2seq/encoder-decoder models
    """
    def initialize(self, memory_bank, src_lengths, src_map=None, device=None,
                   target_prefix=None):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.
        """

        (fn_map_state, memory_bank, src_map,
            target_prefix) = self.initialize_tile(
                memory_bank, src_lengths, src_map, target_prefix)
        # if device is None:
        #     device = self.get_device_from_memory_bank(memory_bank)
        device = None

        super(BeamSearch, self).initialize_(
            memory_bank, self.memory_lengths, src_map, device, target_prefix)

        return fn_map_state, memory_bank, self.memory_lengths, src_map


class EncoderDLSDKModel(CommonDLSDKModel):
    default_model_suffix = 'encoder'
    input_layers = ['src', 'src_len',]
    output_layers = ['memory', 'src_len', 'state.0', 'state.1']

    def fit_to_input(self, input_data):
        if isinstance(input_data, list):
            src = np.array(input_data)
            src_len = np.array([len(input_data),])
            for _ in range(2):
                src = np.expand_dims(src, -1)
            input_data = {'src': src, 'src_len': src_len}
        return super().fit_to_input(input_data)


class DecoderDLSDKModel(CommonDLSDKModel):
    default_model_suffix = 'decoder'
    input_layers = ['c_0', 'h_0', 'input', 'input_feed.1', 'mem_len', 'memory']
    output_layers = ['attn', 'c_1', 'h_1', 'input_feed', 'output']

    hidden_size = 500

    def init_state(self, encoder_final):
        self.state = {}
        # encoder_final = {tuple: 2}(tensor([[[-2.7412e-01, 2.2307e-01, 6.2228e-02, -5.5229e-02, -6.1341e-02,\n
        # 2.0640e-01, 3.3893e-02, 7.4298e-04, -6.4537e-01, -2.9887e-02,\n - 8.3800e-03, 1.2292e-01, 1.2766e-01, -6.1408e-01, 3.0182e-03,\n - 1.8742e-03, 3.1618e-02, -8.0645e-02, 2.2937e-02, 5.5462e-02,\n
        # 4.6654e-01, -1.9155e-01, -9.8324e-02, -5.1246e-01, 2.3397e-01,\n
        # 7.2712e-02, 5.8344e-02, 8.3327e-03, -2.7703e-02, -2.7035e-01,\n
        # 9.6433e-01, -2.6675e-02, -7.3549e-02, -2.6719e-01, -3.3957e-01,\n
        # 6.8648e-04, 1.8804e-02, -1.6088e-02, 2.1493e-01, -8.8760e-02,\n
        # 2.7494e-02, 4.1520e-02, 1.9755e-02, 3.1069e-01, -2.4410e-02,\n
        # 2.0732e-01, -7.9655e-02, 2.9069e-01, 6.2205e-01, 1.5323e-02,\n - 1.4107e-01, 2.3460e-01, -1.7421e-01, -1.0560e-02, -4.7571e-01,\n - 4.4143e-01, 2.2972e-01, 1.3035e-01, -5.7180e-01, 3.4688e-02,\n - 2.5753e-02, 7.3816e-01, -1.1194e-01, -3.1533e-02, -6.5280e-02,\n - 5.6029e-01, -...
        # 0 = {Tensor: 2}
        # tensor([[[-2.7412e-01, 2.2307e-01, 6.2228e-02, -5.5229e-02, -6.1341e-02,\n
        # 2.0640e-01, 3.3893e-02, 7.4298e-04, -6.4537e-01, -2.9887e-02,\n - 8.3800e-03, 1.2292e-01, 1.2766e-01, -6.1408e-01, 3.0182e-03,\n - 1.8742e-03, 3.1618e-02, -8.0645e-02, 2.2937e-02, 5.5462e-02,\n
        # 4.6654e-01, -1.9155e-01, -9.8324e-02, -5.1246e-01, 2.3397e-01,\n
        # 7.2712e-02, 5.8344e-02, 8.3327e-03, -2.7703e-02, -2.7035e-01,\n
        # 9.6433e-01, -2.6675e-02, -7.3549e-02, -2.6719e-01, -3.3957e-01,\n
        # 6.8648e-04, 1.8804e-02, -1.6088e-02, 2.1493e-01, -8.8760e-02,\n
        # 2.7494e-02, 4.1520e-02, 1.9755e-02, 3.1069e-01, -2.4410e-02,\n
        # 2.0732e-01, -7.9655e-02, 2.9069e-01, 6.2205e-01, 1.5323e-02,\n - 1.4107e-01, 2.3460e-01, -1.7421e-01, -1.0560e-02, -4.7571e-01,\n - 4.4143e-01, 2.2972e-01, 1.3035e-01, -5.7180e-01, 3.4688e-02,\n - 2.5753e-02, 7.3816e-01, -1.1194e-01, -3.1533e-02, -6.5280e-02,\n - 5.6029e-01, -1...
        # T = {Tensor: 500}
        # shape = {Size: 3}        torch.Size([2, 1, 500])
        #
        # 1 = {Tensor: 2}
        # torch.Size([2, 1, 500])
        #
        # memory_bank = {Tensor: 7}
        # tensor([[[-0.0101, -0.0095, 0.0026, ..., -0.0137, 0.0043, -0.0049]],\n\n[[-0.0020, 0.0014, 0.0012, ..., 0.0149,
        #                                                                           0.0112, -0.0310]],\n\n[[0.0018,
        #                                                                                                   0.0056,
        #                                                                                                   -0.0004, ...,
        #                                                                                                   -0.0113,
        #                                                                                                   0.0031,
        #                                                                                                   0.0301]],\n\n...,\n\
        # n[[0.0115, 0.0152, 0.0365, ..., 0.0199, 0.0048, 0.0021]],\n\n[[0.0133, 0.0200, 0.0085, ..., 0.1286, 0.0100,
        #                                                                0.0550]],\n\n[
        #     [0.0056, 0.0013, -0.0021, ..., 0.0030, 0.0021, 0.0052]]])
        # T = {Tensor: 500}
        # shape = {Size: 3}        torch.Size([7, 1, 500])
        #
        # src = {Tensor: 7}        tensor([[[0]],\n\n[[6]],\n\n[[2]],\n\n[[16]],\n\n[[3]],\n\n[[7]],\n\n[[4]]])
        # shape = {Size: 3}        torch.Size([7, 1, 1])
        """Initialize decoder state with last state of the encoder."""
        # def _fix_enc_hidden(hidden):
        #     # The encoder hidden is  (layers*directions) x batch x dim.
        #     # We need to convert it to layers x batch x (directions*dim).
        #     if self.bidirectional_encoder:
        #         hidden = torch.cat([hidden[0:hidden.size(0):2],
        #                             hidden[1:hidden.size(0):2]], 2)
        #     return hidden
        #
        # if isinstance(encoder_final, tuple):  # LSTM
        #     self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
        #                                  for enc_hid in encoder_final)
        # else:  # GRU
        #     self.state["hidden"] = (_fix_enc_hidden(encoder_final), )
        self.state["hidden"] = encoder_final

        # Init the input feed.
        batch_size = self.state["hidden"][0].shape[1]
        h_size = (batch_size, self.hidden_size)
        # self.state["input_feed"] = self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["input_feed"] = np.expand_dims(np.zeros(h_size), 0)
        # self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)
        # if self._coverage and self.state["coverage"] is not None:
        #     self.state["coverage"] = fn(self.state["coverage"], 1)

    # def detach_state(self):
    #     self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
    #     self.state["input_feed"] = self.state["input_feed"].detach()

    def predict(self, input, memory_bank, mem_lengths):
        # input_feed = self.model.decoder.state["input_feed"].squeeze(0)
        input_feed = self.state['input_feed'].squeeze(0)

        # input_feed_batch, _ = input_feed.size()
        # _, tgt_batch, _ = decoder_in.size()
        # # aeq(tgt_batch, input_feed_batch)
        # # END Additional args check.

        h_0 = self.state["hidden"][0]
        c_0 = self.state['hidden'][1]

        # coverage = self.model.decoder.state["coverage"].squeeze(0) \
        #     if self.model.decoder.state["coverage"] is not None else None

        # dec_out, dec_attn, dec_hidden, dec_input_feed, dec_coverage = self.model.decoder(
        #     decoder_in, memory_bank, memory_lengths=memory_lengths, step=step,
        #     input_feed=input_feed, hidden=dec_state, coverage=coverage
        # )

        input_data = {'c_0': c_0,
                      'h_0': h_0,
                      'input': input,
                      'input_feed.1': input_feed,
                      'mem_len': mem_lengths,
                      'memory': memory_bank}
        results = self.exec_network.infer(input_data)

        # output_layers = ['attn', 'c_1', 'h_1', 'input_feed', 'output']

        # self.model.decoder.state["hidden"] = dec_hidden
        # self.model.decoder.state["input_feed"] = dec_input_feed

        self.state["hidden"] = (results["h_1"], results["c_1"])
        self.state["input_feed"] = results["input_feed"]

        return results["attn"], results["output"]

class GeneratorDLSDKModel(CommonDLSDKModel):
    default_model_suffix = 'generator'
    input_layers = ['input']
    output_layers = ['output']

    def predict(self, dec_output):
        input_data = {'input': dec_output}
        results = self.exec_network.infer(input_data)
        return results['output']

class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.alpha,
            opt.beta,
            opt.length_penalty,
            opt.coverage_penalty)

    def __init__(self, alpha, beta, length_penalty, coverage_penalty):
        # self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = PenaltyBuilder(coverage_penalty, length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is None or length_penalty == "none":
            if alpha != 0:
                print_info("Non-default `alpha` with no length penalty. `alpha` has no effect.")
        else:
            # using some length penalty
            if length_penalty == "wu" and alpha == 0.:
                print_info("Using length penalty Wu with alpha==0 is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                print_info("Non-default `beta` with no coverage penalty. `beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.:
                print_info("Non-default coverage penalty with beta==0 is equivalent to using coverage penalty none.")

class PenaltyBuilder(object):
    """Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen

    Attributes:
        has_cov_pen (bool): Whether coverage penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting beta
            to 0 should force coverage length to be a no-op.
        has_len_pen (bool): Whether length penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting alpha
            to 1 should force length penalty to be a no-op.
        coverage_penalty (callable[[FloatTensor, float], FloatTensor]):
            Calculates the coverage penalty.
        length_penalty (callable[[int, float], float]): Calculates
            the length penalty.
    """

    def __init__(self, cov_pen, length_pen):
        self.has_cov_pen = not self._pen_is_none(cov_pen)
        self.coverage_penalty = self._coverage_penalty(cov_pen)
        self.has_len_pen = not self._pen_is_none(length_pen)
        self.length_penalty = self._length_penalty(length_pen)

    @staticmethod
    def _pen_is_none(pen):
        return pen == "none" or pen is None

    def _coverage_penalty(self, cov_pen):
        if cov_pen == "wu":
            return self.coverage_wu
        elif cov_pen == "summary":
            return self.coverage_summary
        elif self._pen_is_none(cov_pen):
            return self.coverage_none
        else:
            raise NotImplementedError("No '{:s}' coverage penalty.".format(
                cov_pen))

    def _length_penalty(self, length_pen):
        if length_pen == "wu":
            return self.length_wu
        elif length_pen == "avg":
            return self.length_average
        elif self._pen_is_none(length_pen):
            return self.length_none
        else:
            raise NotImplementedError("No '{:s}' length penalty.".format(
                length_pen))

    # Below are all the different penalty terms implemented so far.
    # Subtract coverage penalty from topk log probs.
    # Divide topk log probs by length penalty.

    def coverage_wu(self, cov, beta=0.):
        """GNMT coverage re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        ``cov`` is expected to be sized ``(*, seq_len)``, where ``*`` is
        probably ``batch_size x beam_size`` but could be several
        dimensions like ``(batch_size, beam_size)``. If ``cov`` is attention,
        then the ``seq_len`` axis probably sums to (almost) 1.
        """

        # penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(-1)
        penalty = -np.min(cov, cov.clone().fill_(1.0)).log().sum(-1)
        return beta * penalty

    def coverage_summary(self, cov, beta=0.):
        """Our summary penalty."""
        # penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
        penalty = np.max(cov, cov.clone().fill_(1.0)).sum(-1)
        penalty -= cov.size(-1)
        return beta * penalty

    def coverage_none(self, cov, beta=0.):
        """Returns zero as penalty"""
        # none = torch.zeros((1,), device=cov.device,
        #                    dtype=torch.float)
        none = np.zeros((1,), dtype=np.float)
        if cov.dim() == 3:
            none = np.expand_dims(none)
        return none

    def length_wu(self, cur_len, alpha=0.):
        """GNMT length re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        return ((5 + cur_len) / 6.0) ** alpha

    def length_average(self, cur_len, alpha=0.):
        """Returns the current sequence length."""
        return cur_len

    def length_none(self, cur_len, alpha=0.):
        """Returns unmodified scores."""
        return 1.0


class CommonONNXModel(BaseModel):
    default_model_suffix = 'encoder'

    def __init__(self, network_info, launcher, *args, **kwargs):
        super().__init__(network_info, launcher)
        model = self.automatic_model_search(network_info)
        self.inference_session = launcher.create_inference_session(str(model))
        self.input_blob = next(iter(self.inference_session.get_inputs()))
        self.output_blob = next(iter(self.inference_session.get_outputs()))

    def predict(self, identifiers, input_data, callback=None):
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
        accepted_suffixes = ['.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_suffix, model))

        return model

class EncoderONNXModel(CommonONNXModel):
    default_model_suffix = 'encoder'
    input_layers = []
    output_layer = []

    def fit_to_input(self, input_data):
        frames, _, _ = input_data.shape
        return {self.input_blob.name: input_data, '1': np.array([frames], dtype=np.int64)}


class DecoderONNXModel(CommonONNXModel):
    default_model_suffix = 'decoder'
    input_layers = ['input.1', '1', '2']
    output_layers = ['151', '152', '153']


class GeneratorONNXModel(CommonONNXModel):
    default_model_suffix = 'generator'
    input_layers = ['0', '1']
    output_layer = []


class DummyModel(BaseModel):
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
