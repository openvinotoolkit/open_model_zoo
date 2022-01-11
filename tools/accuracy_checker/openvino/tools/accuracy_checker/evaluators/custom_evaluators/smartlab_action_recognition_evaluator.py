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
import pickle # nosec - disable B403:import-pickle check
from functools import partial
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import (
    BaseDLSDKModel, BaseCascadeModel, BaseONNXModel, BaseOpenCVModel, BaseOpenVINOModel,
    create_model, create_encoder
)
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, read_pickle, parse_partial_shape


# # Self-define for multi-view inputs
# def extract_image_representations(image_representations, meta_only=False):
#     meta = [rep.metadata for rep in image_representations]
#     print(meta)
#     if meta_only:
#         return meta
#     images = [rep.data for rep in image_representations]

#     return images, meta



class SmarlabActionRecognitionEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)

        # model definition 
        model = SequentialModel(
            config.get('network_info', {}),
            launcher,
            config.get('_models', []),
            config.get('_model_is_blob'),
            delayed_model_loading
        )

        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        # print(self.dataset) # openvino.tools.accuracy_checker.dataset.DataProvider
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)

            # print(self.preprocessor)

            # print('=== preprocessor.process ===')
            # print(batch_inputs)

            batch_inputs_extr, _ = extract_image_representations(batch_inputs)


            # print('=== extract_image_representations ===')
            # print(np.array(batch_inputs_extr).shape) # (1, 224, 224, 3)

            encoder_callback = None
            if output_callback:
                encoder_callback = partial(output_callback, metrics_result=None,
                                            element_identifiers=batch_identifiers,
                                            dataset_indices=batch_input_ids)

            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_inputs_extr, encoder_callback=encoder_callback
            )

            # print('=== smartlab evaluator ===')
            # print(batch_input_ids, batch_prediction)


            # 11Jan2022 dones't work accurayc metric for classfiiction.

            metrics_result = self._get_metrics_result(
                batch_input_ids, batch_annotation, batch_prediction, calculate_metrics)
            metrics_result = {}

            if output_callback:
                output_callback(batch_raw_prediction[0], metrics_result=metrics_result,
                    element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)

            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


    def process_dataset(self, subset=None, num_images=None, check_progress=False, dataset_tag='',
                        output_callback=None, allow_pairwise_subset=False, dump_prediction_to_annotation=False,
                        calculate_metrics=True, **kwargs):

        print("=== process dataset ===")
        # print(self.dataset)
        # print(dataset_tag)
        # print(subset) # None
        self._prepare_dataset(dataset_tag)
        self._create_subset(subset, num_images, allow_pairwise_subset)
        # compute_intermediate_metric_res, metric_interval, ignore_results_formatting, ignore_metric_reference
        metric_config = self.configure_intermediate_metrics_results(kwargs)

        # print(self.dataset.size) # 100
        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )

        self._process(output_callback, calculate_metrics, _progress_reporter, metric_config, kwargs.get('csv_result'))

        if _progress_reporter:
            _progress_reporter.finish()


class SequentialModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['topview_encoder', 'frontview_encoder', 'decoder']

        network_info = self.fill_part_with_model(
            network_info, parts, models_args, is_blob, delayed_model_loading)

        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder and decoder fields')

        self.processing_frames_buffer = []
        self._topview_encoder_mapping = {'dlsdk': EncoderDLSDKModel}
        self._frontview_encoder_mapping = {'dlsdk': EncoderDLSDKModel}
        self._decoder_mapping = {'dlsdk': DecoderDLSDKModel}

        self.topview_encoder = create_encoder(network_info['topview_encoder'],
            launcher, self._topview_encoder_mapping, delayed_model_loading)
        self.frontview_encoder = create_encoder(network_info['frontview_encoder'],
            launcher, self._frontview_encoder_mapping, delayed_model_loading)
        self.decoder = create_model(network_info['decoder'],
            launcher, self._decoder_mapping, 'decoder', delayed_model_loading)

        self._part_by_name = {
            'topview_encoder': self.topview_encoder,
            'frontview_encoder': self.frontview_encoder,
            'decoder': self.decoder}

    def predict(self, identifiers, input_data, encoder_callback=None):
        raw_outputs = []
        predictions = []

        if len(np.shape(input_data)) == 5:
            input_data = input_data[0]

        for data in input_data: # how to get diffent frame input???
            topview_prediction = self.topview_encoder.predict(identifiers, [data])
            frontview_prediction = self.frontview_encoder.predict(identifiers, [data])

            if encoder_callback:
                encoder_callback(topview_prediction)
                encoder_callback(frontview_prediction)

            raw_output, prediction = self.decoder.predict(
                identifiers, [topview_prediction[self.topview_encoder.output_blob]], frontview_prediction[self.topview_encoder.output_blob])
            raw_outputs.append(raw_output)
            predictions.append(prediction)

        return raw_outputs, predictions


class EncoderDLSDKModel(BaseDLSDKModel):
    def predict(self, identifiers, input_data):
        input_dict = self.fit_to_input(input_data)
        # print(input_dict)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: data.shape for key, data in input_dict.items()})
        return self.exec_network.infer(input_dict)

    def fit_to_input(self, input_data):
        # print(np.array(input_data).shape)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info[self.input_blob].input_data
        else:
            input_info = self.exec_network.inputs[self.input_blob]
        if self.input_blob in self.dynamic_inputs or tuple(input_info.shape) != np.shape(input_data):
            self._reshape_input({self.input_blob: np.shape(input_data)})

        return {self.input_blob: np.array(input_data)}


class DecoderDLSDKModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        
        inputs = iter(self.network.input_info)
        self.input_blob1 = next(inputs)
        self.input_blob2 = next(inputs)
        # print(self.input_blob1)
        # print(self.input_blob2)

    def predict(self, identifiers, input_data1, input_data2):
        input_dict = self.fit_to_input(input_data1, input_data2)
        raw_result = self.exec_network.infer(input_dict)

        ### yoclo classifier ###
        isAction = (raw_result[self.output_blob].squeeze()[0] >= .5).astype(int)
        result = isAction*(np.argmax(raw_result[self.output_blob].squeeze()[1:]) + 1)
        # print(isAction)
        # print(result)

        return raw_result, result

    def fit_to_input(self, input_data1, input_data2):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.input_blob].input_data
            if has_info else self.exec_network.inputs[self.input_blob]
        )
        # if not input_info.is_dynamic:
        #     input_data = np.reshape(input_data, input_info.shape)
        return {self.input_blob1: np.array(input_data1), self.input_blob2: np.array(input_data2)}