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

from ..config import ListField, BoolField, ConfigError
from .data_reader import BaseReader, create_ann_identifier_key, AnnotationDataIdentifier
from ..utils import contains_all

class NCFDataReader(BaseReader):
    __provider__ = 'ncf_data_reader'

    def configure(self):
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.data_layout = self.get_value_from_config('data_layout')

    def read(self, data_id):
        if not isinstance(data_id, str):
            raise IndexError('Data identifier must be a string')

        return float(data_id.split(":")[1])


class AnnotationFeaturesReader(BaseReader):
    __provider__ = 'annotation_features_extractor'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({'features': ListField(allow_empty=False, value_type=str, description='List of features.')})
        return parameters

    def configure(self):
        self.feature_list = self.get_value_from_config('features')
        self.single = len(self.feature_list) == 1
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.data_layout = self.get_value_from_config('data_layout')

    def read(self, data_id):
        if isinstance(data_id, AnnotationDataIdentifier):
            ordered_data_id = ['{}_{}'.format(feat, data_id.annotation_id) for feat in self.feature_list]
            data_id.data_id = ordered_data_id if not self.single else ordered_data_id[0]
        relevant_annotation = self.data_source[create_ann_identifier_key(data_id)]
        if not contains_all(relevant_annotation.__dict__, self.feature_list):
            raise ConfigError(
                'annotation_class prototype does not contain provided features {}'.format(', '.join(self.feature_list))
            )
        features = [getattr(relevant_annotation, feature) for feature in self.feature_list]
        if self.single:
            return features[0]
        return features

    def _read_list(self, data_id):
        return self.read(data_id)

    def reset(self):
        self.subset = range(len(self.data_source))
        self.counter = 0


class DiskImageFeaturesExtractor(BaseReader):
    __provider__ = 'disk_features_extractor'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({'input_is_dict_type': BoolField(
            optional=True, default=True, description='Model input is dict type.')})
        parameters.update({'output_is_dict_type': BoolField(
            optional=True, default=True, description='Model output is dict type.')})
        return parameters

    def configure(self):
        self.input_as_dict_type = self.get_value_from_config('input_is_dict_type')
        self.output_is_dict_type = self.get_value_from_config('output_is_dict_type')

    def read(self, data_id):
        assert isinstance(data_id, AnnotationDataIdentifier)
        data = data_id.data_id

        required_keys = ["keypoints", "descriptors", "image_size", "oris"]

        view0 = {
            **{k: data[k + "0"] for k in required_keys if k + "0" in data},
        }
        view1 = {
            **{k: data[k + "1"] for k in required_keys if k + "0" in data},
        }

        return {"image0": view0, "image1": view1}

    def _read_list(self, data_id):
        return self.read(data_id)
