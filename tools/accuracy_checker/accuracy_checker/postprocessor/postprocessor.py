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
from enum import Enum
from ..representation import ContainerRepresentation
from ..config import ConfigValidator, StringField, ConfigError, BaseField
from ..dependency import ClassProvider, UnregisteredProviderException
from ..utils import (
    zipped_transform,
    string_to_list,
    check_representation_type,
    get_supported_representations,
    enum_values,
    get_parameter_value_from_config
)


class Postprocessor(ClassProvider):
    __provider_type__ = 'postprocessor'

    annotation_types = ()
    prediction_types = ()

    @classmethod
    def parameters(cls):
        return {
            'type': StringField(
                default=cls.__provider__ if hasattr(cls, '__provider__') else None, description="Postprocessor type.",
            ),
            'annotation_source': BaseField(
                optional=True,
                description="Annotation identifier in case when complicated representation"
                            " located in representation container is used."
            ),
            'prediction_source': BaseField(
                optional=True,
                description="Output layer name in case when complicated representation"
                            " located in representation container is used."
            )
        }

    def get_value_from_config(self, key):
        return get_parameter_value_from_config(self.config, self.parameters(), key)

    def __init__(self, config, name=None, meta=None, state=None):
        self.config = config
        self.name = name
        self.meta = meta
        self.state = state
        self.image_size = None

        self.annotation_source = self.get_value_from_config('annotation_source')
        if self.annotation_source and not isinstance(self.annotation_source, list):
            self.annotation_source = string_to_list(self.annotation_source)

        self.prediction_source = self.get_value_from_config('prediction_source')
        if self.prediction_source and not isinstance(self.prediction_source, list):
            self.prediction_source = string_to_list(self.prediction_source)

        self.validate_config(config)
        self.setup()

    def __call__(self, *args, **kwargs):
        return self.process_all(*args, **kwargs)

    def setup(self):
        self.configure()

    def process_image(self, annotation, prediction):
        raise NotImplementedError

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        """
        This is the default implementation.
        For the older postprocessor classes that do not know that
        postprocessing may be done with using image_metadata; in this case
        the postprocessor class overrides the method `process_image` only,
        and the overridden method will be called by this default implementation.

        Note that if a postprocessor class wants to use image_metadata,
        it MUST overrides this method `process_image_with_metadata`
        instead of `process_image`.
        """
        return self.process_image(annotation, prediction)

    def process(self, annotation, prediction, image_metadata=None):
        image_size = annotation[0].metadata.get('image_size') if None not in annotation else None
        self.image_size = None
        if image_size:
            self.image_size = image_size[0]
        if self.image_size is None and image_metadata:
            self.image_size = image_metadata.get('image_size')

        self.process_image_with_metadata(annotation, prediction, image_metadata)

        return annotation, prediction

    def process_all(self, annotations, predictions, image_metadata=None):
        assert image_metadata is None, "The method 'process_all' using image_metadata has not been implemented yet"
        zipped_transform(self.process, zipped_transform(self.get_entries, annotations, predictions))
        return annotations, predictions

    def configure(self):
        pass

    @classmethod
    def validate_config(cls, config, fetch_only=False, uri_prefix=''):
        errors = []
        if cls.__name__ == Postprocessor.__name__:
            processing_provider = config.get('type')
            if not processing_provider:
                error = ConfigError('type does not found', config, uri_prefix or 'postprocessing')
                if not fetch_only:
                    raise error
                errors.append(error)
                return errors
            try:
                processor_cls = cls.resolve(processing_provider)
            except UnregisteredProviderException as exception:
                if not fetch_only:
                    raise exception
                errors.append(
                    ConfigError(
                        "postprocessor {} unregistered".format(processing_provider), config,
                        uri_prefix or 'postprocessing', validation_scheme=cls.validation_scheme()
                    )
                )
                return errors
            errors.extend(processor_cls.validate_config(config, fetch_only=fetch_only, uri_prefix=uri_prefix))
            return errors

        postprocessing_uri = uri_prefix or 'postprocessing.{}'.format(cls.__provider__)
        return ConfigValidator(
            postprocessing_uri, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT, fields=cls.parameters()
        ).validate(config, fetch_only=fetch_only, validation_scheme=cls.validation_scheme())

    def get_entries(self, annotation, prediction):
        message_not_found = '{}: {} is not found in container'
        message_incorrect_type = "Incorrect type of {}. Postprocessor {} can work only with {}"

        def resolve_container(container, supported_types, entry_name, sources=None):
            if not isinstance(container, ContainerRepresentation):
                if sources:
                    message = 'Warning: {}_source can be applied only to container. Default value will be used'
                    warnings.warn(message.format(entry_name))

                return [container]

            if not sources:
                return get_supported_representations(container.values(), supported_types)

            entries = []
            for source in sources:
                representation = container.get(source)
                if not representation:
                    raise ConfigError(message_not_found.format(entry_name, source))

                if supported_types and not check_representation_type(representation, supported_types):
                    raise TypeError(message_incorrect_type.format(entry_name, self.name, ','.join(supported_types)))

                entries.append(representation)

            return entries

        annotation_entries = resolve_container(annotation, self.annotation_types, 'annotation', self.annotation_source)
        prediction_entries = resolve_container(prediction, self.prediction_types, 'prediction', self.prediction_source)

        return annotation_entries, prediction_entries

    @classmethod
    def validation_scheme(cls, provider=None):
        if cls.__name__ == Postprocessor.__name__:
            if provider:
                return cls.resolve(provider).validation_scheme()
            full_scheme = []
            for provider_ in cls.providers:
                full_scheme.append(cls.resolve(provider_).validation_scheme())
            return full_scheme
        return cls.parameters()


class ApplyToOption(Enum):
    ANNOTATION = 'annotation'
    PREDICTION = 'prediction'
    ALL = 'all'


class PostprocessorWithSpecificTargets(Postprocessor):
    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'apply_to': StringField(
                optional=True, choices=enum_values(ApplyToOption),
                description="determines target boxes for processing (annotation for ground truth "
                            "boxes and prediction for detection results, all for both)."
            )
        })
        return parameters

    def setup(self):
        apply_to = self.get_value_from_config('apply_to')
        self._required_both = False
        self.apply_to = ApplyToOption(apply_to) if apply_to else None
        self._deprocess_predictions = False

        if (self.annotation_source or self.prediction_source) and self.apply_to:
            raise ConfigError("apply_to and sources both provided. You need specify only one from them")

        if not self.annotation_source and not self.prediction_source and not self.apply_to:
            raise ConfigError("apply_to or annotation_source or prediction_source required for {}".format(self.name))

        self.configure()

    def process(self, annotation, prediction, image_metadata=None):
        image_size = annotation[0].metadata.get('image_size') if None not in annotation else None
        self.image_size = None
        if image_size:
            self.image_size = image_size[0]
        if self.image_size is None and image_metadata:
            self.image_size = image_metadata.get('image_size')
        target_annotations, target_predictions = None, None
        if self.annotation_source or self.prediction_source:
            target_annotations, target_predictions = self._choose_targets_using_sources(annotation, prediction)

        if self.apply_to:
            target_annotations, target_predictions = self._choose_targets_using_apply_to(annotation, prediction)

        if not target_annotations and not target_predictions:
            raise ValueError("Suitable targets for {} not found".format(self.name))

        self.process_image_with_metadata(target_annotations, target_predictions, image_metadata)

        return annotation, prediction

    def _choose_targets_using_sources(self, annotations, predictions):
        target_annotations = annotations if self.annotation_source else []
        target_predictions = predictions if self.prediction_source else []

        return target_annotations, target_predictions

    def _choose_targets_using_apply_to(self, annotations, predictions):
        if all(annotation is None for annotation in annotations):
            apply_to = ApplyToOption.PREDICTION
            self._deprocess_predictions = True
        else:
            apply_to = self.apply_to if not self._required_both else ApplyToOption.ALL
        targets_specification = {
            ApplyToOption.ANNOTATION: (annotations, []),
            ApplyToOption.PREDICTION: ([], predictions),
            ApplyToOption.ALL: (annotations, predictions)
        }

        return targets_specification[apply_to]

    def process_image(self, annotation, prediction):
        raise NotImplementedError
