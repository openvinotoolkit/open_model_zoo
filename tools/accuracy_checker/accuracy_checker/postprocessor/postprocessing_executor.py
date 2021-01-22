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

from ..config import ConfigValidator, StringField
from ..utils import overrides, zipped_transform
from .postprocessor import Postprocessor


class PostprocessingExecutor:
    def __init__(self, processors=None, dataset_name='custom', dataset_meta=None, state=None):
        self._processors = []
        self._image_processors = []
        self._dataset_processors = []
        self.dataset_meta = dataset_meta

        self.state = state or {}

        self.allow_image_postprocessor = True

        if not processors:
            return

        for config in processors:
            postprocessor_config = PostprocessorConfig(
                "{}.postprocessing".format(dataset_name),
                on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            postprocessor_config.validate(config)
            self.register_postprocessor(config)

    def process_dataset(self, annotations, predictions):
        for method in self._dataset_processors:
            annotations, predictions = method.process_all(annotations, predictions)

        return annotations, predictions

    def process_image(self, annotation, prediction, image_metadata=None):
        for method in self._image_processors:
            annotation_entries, prediction_entries = method.get_entries(annotation, prediction)
            method.process(annotation_entries, prediction_entries, image_metadata)

        return annotation, prediction

    def process_batch(self, annotations, predictions, metas=None, allow_empty_annotation=False):
        if allow_empty_annotation and not annotations:
            annotations = [None] * len(predictions)
        # FIX IT: remove zipped_transform here in the future -- it is too flexible and unpredictable
        if metas is None:
            zipped_result = zipped_transform(self.process_image, annotations, predictions)
        else:
            zipped_result = zipped_transform(self.process_image, annotations, predictions, metas)

        return zipped_result[0:2]  # return changed annotations and predictions only

    def full_process(self, annotations, predictions, metas=None):
        return self.process_dataset(*self.process_batch(annotations, predictions, metas))

    @property
    def has_dataset_processors(self):
        return len(self._dataset_processors) != 0

    @property
    def has_processors(self):
        return len(self._image_processors) + len(self._dataset_processors) != 0

    def __call__(self, context, *args, **kwargs):
        batch_annotation = context.annotation_batch
        batch_prediction = context.prediction_batch
        batch_meta = getattr(context, 'meta_batch', None) # context could have meta_batch in the future
        context.batch_annotation, context.batch_prediction = self.process_batch(batch_annotation,
                                                                                batch_prediction,
                                                                                batch_meta)

    def register_postprocessor(self, config):
        postprocessor = Postprocessor.provide(config['type'], config, config['type'], self.dataset_meta, self.state)
        self._processors.append(postprocessor)
        if overrides(postprocessor, 'process_all', Postprocessor):
            self.allow_image_postprocessor = False
            self._dataset_processors.append(postprocessor)
            return
        if self.allow_image_postprocessor:
            self._image_processors.append(postprocessor)
        else:
            self._dataset_processors.append(postprocessor)

    @classmethod
    def validate_config(cls, processors, fetch_only=False, uri_prefix=''):
        if not processors:
            return []
        errors = []
        for processor_id, processor in enumerate(processors):
            processor_uri = '{}.{}'.format(uri_prefix or 'postprocessing', processor_id)
            errors.extend(Postprocessor.validate_config(processor, fetch_only=fetch_only, uri_prefix=processor_uri))

        return errors


class PostprocessorConfig(ConfigValidator):
    type = StringField(choices=Postprocessor.providers)
