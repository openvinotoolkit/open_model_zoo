"""
 Copyright (c) 2018 Intel Corporation

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
from enum import Enum
from ..representation import ContainerRepresentation
from ..config import ConfigValidator, StringField, ConfigError, BaseField
from ..dependency import ClassProvider
from ..utils import (zipped_transform, string_to_list, check_representation_type, get_supported_representations,
                     enum_values)


class BasePostprocessorConfig(ConfigValidator):
    type = StringField()
    annotation_source = BaseField(optional=True)
    prediction_source = BaseField(optional=True)


class Postprocessor(ClassProvider):
    __provider_type__ = 'postprocessor'
    annotation_types = ()
    prediction_types = ()

    def __init__(self, config, name=None, meta=None, state=None):
        self.config = config
        self.name = name
        self.meta = meta
        self.state = state
        self.image_size = None
        self.annotation_source = self.config.get('annotation_source')
        if self.annotation_source is not None and not isinstance(self.annotation_source, list):
            self.annotation_source = string_to_list(self.annotation_source)
        self.prediction_source = self.config.get('prediction_source')
        if self.prediction_source is not None and not isinstance(self.prediction_source, list):
            self.prediction_source = string_to_list(self.prediction_source)

        self.validate_config()
        self.setup()

    def __call__(self, *args, **kwargs):
        return self.process_all(*args, **kwargs)

    def setup(self):
        self.configure()

    def process_image(self, annotation, prediction):
        raise NotImplementedError

    def process(self, annotation, prediction):
        self.image_size = annotation[0].metadata.get('image_size') if annotation else None
        self.process_image(annotation, prediction)
        return annotation, prediction

    def process_all(self, annotations, predictions):
        zipped_transform(self.process, zipped_transform(self.get_entries, annotations, predictions))
        return annotations, predictions

    def configure(self):
        pass

    def validate_config(self):
        BasePostprocessorConfig(self.name).validate(self.config)

    def get_entries(self, annotation, prediction):
        message_not_found = '{}: {} is not found in container'
        message_incorrect_type = "Incorrect type of {}. Postprocessor {} can work only with {}"

        def resolve_container(container, supported_types, entry_name, sources=None):
            if not isinstance(container, ContainerRepresentation):
                if sources is not None:
                    warnings.warn('Warning: {}_source can be applied only to container. default {} will be used'.format(
                        entry_name, entry_name))
                return [container]
            if not sources:
                return get_supported_representations(container.values(), supported_types)
            entries = []
            for source in sources:
                rep = container.get(source)

                if not rep:
                    raise ConfigError(message_not_found.format(entry_name, source))

                if supported_types and not check_representation_type(rep, supported_types):
                    raise TypeError(message_incorrect_type.format(entry_name, self.name, ','.join(supported_types)))
                entries.append(rep)
            return entries

        annotation_entries = resolve_container(annotation, self.annotation_types, 'annotation', self.annotation_source)
        prediction_entries = resolve_container(prediction, self.prediction_types, 'prediction', self.prediction_source)

        return annotation_entries, prediction_entries


class ApplyToOption(Enum):
    ANNOTATION = 'annotation'
    PREDICTION = 'prediction'
    ALL = 'all'


class PostprocessorWithTargetsConfigValidator(BasePostprocessorConfig):
    apply_to = StringField(optional=True, choices=enum_values(ApplyToOption))


class PostprocessorWithSpecificTargets(Postprocessor):
    def validate_config(self):
        _config_validator = PostprocessorWithTargetsConfigValidator(self.__provider__)
        _config_validator.validate(self.config)

    def setup(self):
        apply_to_ = self.config.get('apply_to')
        self.apply_to = ApplyToOption(apply_to_) if apply_to_ is not None else None
        if (self.annotation_source is not None or self.prediction_source is not None) and self.apply_to is not None:
            raise ConfigError("apply_to and sources both provided. You need specify only one from them")
        if self.annotation_source is None and self.prediction_source is None and self.apply_to is None:
            raise ConfigError("apply_to or annotation_source or prediction_source required for {}".format(self.name))
        self.configure()

    def process(self, annotation, prediction):
        self.image_size = annotation[0].metadata.get('image_size') if not None in annotation else None
        target_annotations, target_predictions = None, None
        if self.annotation_source is not None or self.prediction_source is not None:
            target_annotations, target_predictions = self._choise_targets_using_sources(annotation, prediction)

        if self.apply_to is not None:
            target_annotations, target_predictions = self._choise_targets_using_apply_to(annotation, prediction)
        if not target_annotations and not target_predictions:
            raise ValueError("Suitable targets for {} not found".format(self.name))
        self.process_image(target_annotations, target_predictions)
        return annotation, prediction

    def _choise_targets_using_sources(self, annotations, predictions):
        target_annotations = []
        target_predictions = []
        if self.annotation_source is not None:
            target_annotations = annotations
        if self.prediction_source is not None:
            target_predictions = predictions
        return target_annotations, target_predictions

    def _choise_targets_using_apply_to(self, annotations, predictions):
        targets_specification = {ApplyToOption.ANNOTATION: (annotations, []),
                                 ApplyToOption.PREDICTION: ([], predictions),
                                 ApplyToOption.ALL: (annotations, predictions)}
        return targets_specification[self.apply_to]

    def process_image(self, annotation, prediction):
        raise NotImplementedError
