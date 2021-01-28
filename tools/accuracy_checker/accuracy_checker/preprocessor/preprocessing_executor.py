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

import warnings
from ..config import ConfigValidator, StringField
from .preprocessor import Preprocessor, MULTI_INFER_PREPROCESSORS
from .ie_preprocessor import IEPreprocessor, ie_preprocess_available


class PreprocessingExecutor:
    def __init__(
            self, processors=None, dataset_name='custom', dataset_meta=None,
            input_shapes=None,
            enable_ie_preprocessing=False
    ):
        self.processors = []
        self.dataset_meta = dataset_meta
        self._multi_infer_transformations = False
        self.ie_processor = None
        if enable_ie_preprocessing:
            if not ie_preprocess_available():
                warnings.warn(
                    'PreProcessInfo is not available in your InferenceEngine version or openvino is not installed'
                    '--ie_preprocessing key will be ignored'
                )
            else:
                self.ie_processor = IEPreprocessor(processors)
                processors = self.ie_processor.keep_preprocessing_info

        if not processors:
            return

        identifier = 'type'
        for processor in processors:
            preprocessor_config = PreprocessorConfig(
                "{}.preprocessors".format(dataset_name), on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
            )

            type_ = processor.get(identifier)
            preprocessor_config.validate(processor, type_)
            preprocessor = Preprocessor.provide(
                processor[identifier], config=processor, name=type_
            )
            if processor[identifier] in MULTI_INFER_PREPROCESSORS:
                self._multi_infer_transformations = True

            self.processors.append(preprocessor)

        if input_shapes is not None:
            self.input_shapes = input_shapes

    def __call__(self, context, *args, **kwargs):
        batch_data = context.data_batch
        batch_annotation = context.annotation_batch
        context.data_batch = self.process(batch_data, batch_annotation)

    def process(self, images, batch_annotation=None):
        for i, _ in enumerate(images):
            for processor in self.processors:
                images[i] = processor(
                    image=images[i], annotation_meta=batch_annotation[i].metadata if batch_annotation else None
                )

        return images

    @property
    def has_multi_infer_transformations(self):
        return self._multi_infer_transformations

    @property
    def input_shapes(self):
        return self._input_shapes

    @input_shapes.setter
    def input_shapes(self, input_shapes):
        self._input_shapes = input_shapes
        for preprocessor in self.processors:
            preprocessor.set_input_shape(input_shapes)

    @property
    def preprocess_info(self):
        if not self.ie_processor:
            return None
        return self.ie_processor.preprocess_info

    @property
    def ie_preprocess_steps(self):
        if not self.ie_processor:
            return []
        return self.ie_processor.steps

    @classmethod
    def validate_config(cls, processors, fetch_only=False, uri_prefix=''):
        if not processors:
            return []
        errors = []
        for preprocessor_id, processor in enumerate(processors):
            preprocessor_uri = '{}.{}'.format(uri_prefix or 'preprocessing', preprocessor_id)
            errors.extend(Preprocessor.validate_config(processor, fetch_only=fetch_only, uri_prefix=preprocessor_uri))

        return errors


class PreprocessorConfig(ConfigValidator):
    type = StringField(choices=Preprocessor.providers)
