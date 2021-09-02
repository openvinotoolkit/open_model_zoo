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

import os, math
from pathlib import Path
import numpy as np

from ..representation import ClassificationAnnotation
from ..config import NumberField, StringField, PathField, BoolField, ConfigError
from .format_converter import BaseFormatConverter
from .format_converter import ConverterReturn

class CriteoKaggleDACConverter(BaseFormatConverter):

    __provider__ = 'criteo'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            "batch": NumberField(optional=True, default=128, description="Model batch"),
            "max_ind_range": NumberField(optional=True, default=0, description="Maximum index range"),
            "subsample_size": NumberField(optional=True, default=0,
                                          description="Limit total record count to batch * subsample size"),
            "validation": BoolField(optional=True, default=True,
                                    description="Allows to use half of dataset for validation purposes"),
            "block": BoolField(optional=True, default=True,
                               description="Make batch-oriented annotations"),
            "separator": StringField(optional=True, default='#',
                                     description="Separator between input identifier and file identifier"),
            "preprocessed_dir": PathField(optional=False, is_directory=True, check_exists=True,
                                          description="Preprocessed dataset location"),
            "dense_features": StringField(optional=True, default='input.1',
                                          description="Name of model dense features input"),
            "sparse_features": StringField(optional=True, default='lS_i',
                                           description="Name of model sparse features input. " +
                                           "For multiple inputs use comma-separated list in form <name>:<index>"),
            "lso_features": StringField(optional=True, default='lS_o', description="Name of lS_o-like features input."),
            "save_preprocessed_features": BoolField(
                optional=True, default=True, description='Save preprocessed features or not'
            )
        })

        return parameters

    def configure(self):
        self.src = self.get_value_from_config('testing_file')
        self.batch = int(self.get_value_from_config('batch'))
        self.max_ind_range = int(self.get_value_from_config('max_ind_range'))
        self.subsample = int(self.get_value_from_config('subsample_size'))
        self.validation = self.get_value_from_config('validation')
        self.block = self.get_value_from_config('block')
        self.separator = self.get_value_from_config('separator')
        self.preprocessed_dir = self.get_value_from_config('preprocessed_dir')
        self.dense_features = self.get_value_from_config('dense_features')
        self.sparse_features = self.get_value_from_config('sparse_features')
        self.lso_features = self.get_value_from_config('lso_features')
        self.save_preprocessed_features = self.get_value_from_config('save_preprocessed_features')
        self.parse_sparse_features()

    def parse_sparse_features(self):
        features = self.sparse_features.split(',')
        if len(features) == 1:
            self.sparse_features = {features[0]: range(26)}
        else:
            self.sparse_features = {}
            for feat in features:
                parts = feat.split(':')
                if len(parts) == 2:
                    self.sparse_features[parts[0]] = int(parts[1])
                else:
                    ConfigError('Invalid configuration option {}'.format(feat))

    def convert(self, check_content=False, **kwargs):
        preprocessed_folder = Path(self.preprocessed_dir)
        input_folder = preprocessed_folder / "bs{}".format(self.batch) / 'input'

        if not input_folder.exists() and self.save_preprocessed_features:
            input_folder.mkdir(parents=True)

        annotations = []

        subfolder = 0
        filecnt = 0

        data = np.load(self.src)
        x_int = data['X_int']
        x_cat = data['X_cat']
        y = data['y']
        samples, cat_feat = x_cat.shape

        samples = (samples // self.batch) * self.batch
        start = 0

        if self.subsample:
            samples = self.subsample * self.batch if samples > self.subsample * self.batch else samples
        elif self.validation:
            start = samples // 2

        for i in range(start, samples - self.batch + 1, self.batch):
            c_input = input_folder / "{:02d}".format(subfolder)
            c_input = c_input / "{:06d}.npz".format(i)

            if self.save_preprocessed_features:
                if not c_input.parent.exists():
                    c_input.parent.mkdir(parents=True)

                sample = {
                    self.dense_features: np.log1p(x_int[i:i+self.batch, ...]),
                    self.lso_features: np.dot(np.expand_dims(np.linspace(0, self.batch - 1, num=self.batch), -1),
                                              np.ones((1, cat_feat))).T
                }

                for name in self.sparse_features.keys():
                    x_cat_batch = x_cat[i:i+self.batch, self.sparse_features[name]]
                    x_cat_batch = x_cat_batch % self.max_ind_range if self.max_ind_range > 0 else x_cat_batch
                    sample[name] = x_cat_batch.T

                np.savez_compressed(str(c_input), **sample)

            filecnt += 1
            filecnt %= 0x100

            subfolder = subfolder + 1 if filecnt == 0 else subfolder

            c_file = str(c_input.relative_to(preprocessed_folder))

            if self.block:
                identifiers = [
                    "{}_{}{}{}".format(self.dense_features, i, self.separator, c_file),
                    "{}_{}{}{}".format(self.lso_features, i, self.separator, c_file),
                ]
                for name in self.sparse_features.keys():
                    identifiers.append("{}_{}{}{}".format(name, i, self.separator, c_file))
                annotations.append(ClassificationAnnotation(identifiers, y[i:i+self.batch, ...]))
            else:
                for j in range(i, i + self.batch):
                    identifiers = [
                        "{}_{}{}{}".format(self.dense_features, j, self.separator, c_file),
                        "{}_{}{}{}".format(self.lso_features, j, self.separator, c_file),
                    ]
                    for name in self.sparse_features.keys():
                        identifiers.append("{}_{}{}{}".format(name, j, self.separator, c_file))
                    annotations.append(ClassificationAnnotation(identifiers, y[j, ...]))

        return ConverterReturn(annotations, None, None)

class CriteoBinaryConverter(BaseFormatConverter):

    __provider__ = 'criteo_binary'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            "batch": NumberField(optional=True, default=128, description="Model batch"),
            "max_ind_range": NumberField(optional=True, default=0, description="Maximum index range"),
            "subsample_size": NumberField(optional=True, default=0,
                                          description="Limit total record count to batch * subsample size"),
            "validation": BoolField(optional=True, default=True,
                                    description="Allows to use half of dataset for validation purposes"),
            "block": BoolField(optional=True, default=True,
                               description="Make batch-oriented annotations"),
            "separator": StringField(optional=True, default='#',
                                     description="Separator between input identifier and file identifier"),
            "preprocessed_dir": PathField(optional=False, is_directory=True, check_exists=True,
                                          description="Preprocessed dataset location"),
            "dense_features": StringField(optional=True, default='input.1',
                                          description="Name of model dense features input"),
            "sparse_features": StringField(optional=True, default='lS_i',
                                           description="Name of model sparse features input. " +
                                           "For multiple inputs use comma-separated list in form <name>:<index>"),
            "lso_features": StringField(optional=True, default='lS_o', description="Name of lS_o-like features input."),
            "save_preprocessed_features": BoolField(
                optional=True, default=True, description='Save preprocessed features or not'
            )
        })

        return parameters

    def configure(self):
        self.src = self.get_value_from_config('testing_file')
        self.batch = int(self.get_value_from_config('batch'))
        self.max_ind_range = int(self.get_value_from_config('max_ind_range'))
        self.subsample = int(self.get_value_from_config('subsample_size'))
        self.validation = self.get_value_from_config('validation')
        self.block = self.get_value_from_config('block')
        self.separator = self.get_value_from_config('separator')
        self.preprocessed_dir = self.get_value_from_config('preprocessed_dir')
        self.dense_features = self.get_value_from_config('dense_features')
        self.sparse_features = self.get_value_from_config('sparse_features')
        self.lso_features = self.get_value_from_config('lso_features')
        self.save_preprocessed_features = self.get_value_from_config('save_preprocessed_features')
        self.parse_sparse_features()

    def parse_sparse_features(self):
        features = self.sparse_features.split(',')
        if len(features) == 1:
            self.sparse_features = {features[0]: range(26)}
        else:
            self.sparse_features = {}
            for feat in features:
                parts = feat.split(':')
                if len(parts) == 2:
                    self.sparse_features[parts[0]] = int(parts[1])
                else:
                    ConfigError('Invalid configuration option {}'.format(feat))

    def convert(self, check_content=False, **kwargs):
        preprocessed_folder = Path(self.preprocessed_dir)
        input_folder = preprocessed_folder / "bs{}".format(self.batch) / 'input'

        if not input_folder.exists() and self.save_preprocessed_features:
            input_folder.mkdir(parents=True)

        annotations = []

        subfolder = 0
        filecnt = 0

        bytes_per_feature = 4
        tar_fea = 1   # single target
        den_fea = 13  # 13 dense  features
        spa_fea = 26  # 26 sparse features
        tad_fea = tar_fea + den_fea
        tot_fea = tad_fea + spa_fea
        bytes_per_entry = bytes_per_feature * tot_fea * self.batch
        fea_shape = (self.batch, tot_fea)

        samples = math.ceil(os.path.getsize(self.src) / bytes_per_entry)

        binfile = open(self.src, 'rb') if self.save_preprocessed_features else None

        start = 0

        if self.subsample:
            samples = self.subsample if samples > self.subsample else samples
        elif self.validation:
            start = samples // 2

        for i in range(start, samples, 1):
            c_input = input_folder / "{:02d}".format(subfolder)
            c_input = c_input / "{:06d}.npz".format(i)

            if self.save_preprocessed_features:
                if not c_input.parent.exists():
                    c_input.parent.mkdir(parents=True)

                binfile.seek(i * bytes_per_entry, 0)
                raw_data = binfile.read(bytes_per_entry)
                array = np.frombuffer(raw_data, dtype=np.int32).reshape(fea_shape)
                x_int = array[:, 1:14]
                x_cat = array[:, 14:]
                y = array[:, 0]

                sample = {
                    self.dense_features: np.log1p(x_int),
                    self.lso_features: np.dot(np.expand_dims(np.linspace(0, self.batch - 1, num=self.batch), -1),
                                              np.ones((1, spa_fea))).T
                }

                for name in self.sparse_features.keys():
                    x_cat = x_cat % self.max_ind_range if self.max_ind_range > 0 else x_cat
                    sample[name] = x_cat.T

                np.savez_compressed(str(c_input), **sample)

            filecnt += 1
            filecnt %= 0x100

            subfolder = subfolder + 1 if filecnt == 0 else subfolder

            c_file = str(c_input.relative_to(preprocessed_folder))

            if self.block:
                identifiers = [
                    "{}_{}{}{}".format(self.dense_features, i, self.separator, c_file),
                    "{}_{}{}{}".format(self.lso_features, i, self.separator, c_file),
                ]
                for name in self.sparse_features.keys():
                    identifiers.append("{}_{}{}{}".format(name, i, self.separator, c_file))
                annotations.append(ClassificationAnnotation(identifiers, y))
            else:
                for j in range(self.batch):
                    identifiers = [
                        "{}_{}{}{}".format(self.dense_features, j, self.separator, c_file),
                        "{}_{}{}{}".format(self.lso_features, j, self.separator, c_file),
                    ]
                    for name in self.sparse_features.keys():
                        identifiers.append("{}_{}{}{}".format(name, j, self.separator, c_file))
                    annotations.append(ClassificationAnnotation(identifiers, y[j, ...]))

        if self.save_preprocessed_features:
            binfile.close()

        return ConverterReturn(annotations, None, None)
