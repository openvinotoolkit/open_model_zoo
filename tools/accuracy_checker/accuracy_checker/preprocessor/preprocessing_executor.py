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

import warnings
from ..config import ConfigValidator, StringField
from ..utils import get_data_shapes
from .preprocessor import Preprocessor, MULTI_INFER_PREPROCESSORS
from .launcher_preprocessing import preprocessing_available, get_preprocessor


class PreprocessingExecutor:
    def __init__(
            self, processors=None, dataset_name='custom', dataset_meta=None,
            input_shapes=None, enable_runtime_preprocessing=False, runtime_framework=None
    ):
        self.processors = []
        self.dataset_meta = dataset_meta
        self._multi_infer_transformations = False
        self.ie_processor = None
        if enable_runtime_preprocessing:
            if not preprocessing_available(runtime_framework):
                warnings.warn(
                    f'Preprocessing for {runtime_framework} is not available, '
                    'specified in command line parameter will be ignored')
            else:
                self.ie_processor = get_preprocessor(runtime_framework)(processors)
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
            if processor[identifier] in MULTI_INFER_PREPROCESSORS or getattr(preprocessor, 'to_multi_infer', False):
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

    @property
    def dynamic_shapes(self):
        shape_modification = []
        for processor in self.processors:
            if not processor.shape_modificator:
                continue
            shape_modification.append(processor.dynamic_result_shape)
        if not shape_modification:
            return False
        return shape_modification[-1]

    @property
    def has_shape_modifications(self):
        for processor in self.processors:
            if processor.shape_modificator:
                return True
        return False

    def query_shapes(self, data_shape):
        for processor in self.processors:
            data_shape = processor.query_shapes(data_shape)
        return data_shape

    def query_data_batch_shapes(self, data):
        shapes = []
        for input_data in data:
            shapes.append(self.query_shapes(get_data_shapes(input_data)))
        return shapes


class PreprocessorConfig(ConfigValidator):
    type = StringField(choices=Preprocessor.providers)
