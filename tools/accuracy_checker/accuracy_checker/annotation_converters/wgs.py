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

from pathlib import Path
import numpy as np

from ..representation import ClassificationAnnotation
from ..config import PathField, BoolField
from ..utils import UnsupportedPackage, read_pickle
from .format_converter import BaseFormatConverter, ConverterReturn


class WGSTFRecords(BaseFormatConverter):
    __provider__ = 'wgs_tf_records'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotation_file': PathField(description='path to Deepvariant WGS preprocessed dataset file'),
            "preprocessed_dir": PathField(optional=False, is_directory=True, check_exists=True,
                                          description="Preprocessed dataset location"),
            "skip_dump": BoolField(optional=True, default=True, description='Annotate without saving features')
        })

        return parameters

    def configure(self):
        try:
            import tensorflow as tf # pylint: disable=C0415
            self.tf = tf
        except ImportError as import_error:
            UnsupportedPackage("tf", import_error.msg).raise_error(self.__provider__)
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.preprocessed_dir = self.get_value_from_config('preprocessed_dir')
        self.data_dir = self.get_value_from_config('data_dir')
        self.skip_dump = self.get_value_from_config('skip_dump')

    def read_records(self):
        try:
            record_iterator = self.tf.python_io.tf_record_iterator(path=str(self.annotation_file))
        except AttributeError:
            record_iterator = self.tf.compat.v1.python_io.tf_record_iterator(path=str(self.annotation_file))
        record_list = []
        for string_record in record_iterator:
            example = self.tf.train.Example()
            example.ParseFromString(string_record)
            image = np.frombuffer(example.features.feature['image/encoded'].bytes_list.value[0], dtype=np.uint8)
            image_shape = example.features.feature['image/shape'].int64_list.value
            sequencing_type = example.features.feature['sequencing_type'].int64_list.value
            variant_type = example.features.feature['variant_type'].int64_list.value
            alt_allele_indices_encoded = example.features.feature['alt_allele_indices/encoded'].bytes_list.value
            variant_encoded = example.features.feature['variant/encoded'].bytes_list.value
            label = example.features.feature['label'].int64_list.value
            locus = example.features.feature['locus'].bytes_list.value
            image = np.reshape(image, list(image_shape))
            record_list.append((image,
                                list(label),
                                locus[0].decode('utf-8'),
                                list(sequencing_type),
                                list(variant_type),
                                variant_encoded[0],
                                alt_allele_indices_encoded[0]))
        return record_list

    def convert(self, check_content=False, **kwargs):
        examples = self.read_records()
        annotations = []
        preprocessed_folder = Path(self.preprocessed_dir)
        if not self.skip_dump and not preprocessed_folder.exists():
            preprocessed_folder.mkdir(exist_ok=True, parents=True)
        for iteration, example in enumerate(examples):
            image = example[0]
            label = example[1]
            c_input = preprocessed_folder / "{:06d}.npy".format(iteration)
            if not self.skip_dump:
                np.save(str(c_input), image)
            annotations.append(ClassificationAnnotation(str(c_input.relative_to(preprocessed_folder)), label))
        return ConverterReturn(annotations, None, None)


class WGSPickleRecords(WGSTFRecords):
    __provider__ = 'wgs_pickle_records'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotation_file': PathField(description='path to Deepvariant WGS preprocessed dataset file'),
            "preprocessed_dir": PathField(optional=False, is_directory=True, check_exists=True,
                                          description="Preprocessed dataset location"),
            "skip_dump": BoolField(optional=True, default=True, description='Annotate without saving features')
        })

        return parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.preprocessed_dir = self.get_value_from_config('preprocessed_dir')
        self.data_dir = self.get_value_from_config('data_dir')
        self.skip_dump = self.get_value_from_config('skip_dump')

    def read_records(self):
        return read_pickle(self.annotation_file)
