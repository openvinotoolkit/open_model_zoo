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
            "binary": BoolField(optional=True, default=False,
                                description="Allows input file in binary mode instead of .npz mode"),
            "batch": NumberField(optional=True, default=128, description="Model batch"),
            "max_ind_range": NumberField(optional=True, default=None, value_type=int, min_value=1,
                                         description="Maximum index range for categorical features"),
            "subsample_size": NumberField(optional=True, default=0,
                                          description="Limit total record count to batch * subsample size"),
            "validation": BoolField(optional=True, default=False,
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
        self.binary = self.get_value_from_config('binary')
        self.batch = int(self.get_value_from_config('batch'))
        max_ind_range = self.get_value_from_config('max_ind_range')
        self.max_ind_range = int(max_ind_range) if max_ind_range is not None else max_ind_range
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

    def load_data_file(self):
        if self.binary:
            bytes_per_feature = 4
            tar_fea = 1  # single target
            den_fea = 13  # 13 dense  features
            spa_fea = 26  # 26 sparse features
            tad_fea = tar_fea + den_fea
            tot_fea = tad_fea + spa_fea
            self._bytes_per_entry = bytes_per_feature * tot_fea * self.batch
            self._fea_shape = (self.batch, tot_fea)

            self.count = np.ceil(self.src.stat().st_size / self._bytes_per_entry)
            self.cat_feat = spa_fea

            self.binfile = open(self.src, 'rb') if self.save_preprocessed_features else None # pylint: disable=R1732
        else:
            data = np.load(self.src)
            self._x_int = data['X_int']
            self._x_cat = data['X_cat']
            self._y = data['y']
            self.count, self.cat_feat = self._x_cat.shape
            self.count = np.ceil(self.count / self.batch)

    def close_data_file(self):
        if self.binary:
            if self.save_preprocessed_features:
                self.binfile.close()

    def get_data(self, step):
        if self.binary:
            self.binfile.seek(step * self._bytes_per_entry, 0)
            raw_data = self.binfile.read(self._bytes_per_entry)
            array = np.frombuffer(raw_data, dtype=np.int32).reshape(self._fea_shape)
            return array[:, 1:14], array[:, 14:], array[:, 0]
        start = step * self.batch
        x_int_batch = self._x_int[start:start + self.batch, ...]
        x_cat_batch = self._x_cat[start:start + self.batch, ...]
        y_batch = self._y[start:start + self.batch, ...]
        return x_int_batch, x_cat_batch, y_batch

    def convert(self, check_content=False, **kwargs):
        preprocessed_folder = Path(self.preprocessed_dir)
        input_folder = preprocessed_folder / "bs{}".format(self.batch) / 'input'

        if not input_folder.exists() and self.save_preprocessed_features:
            input_folder.mkdir(parents=True)

        annotations = []

        subfolder = 0
        filecnt = 0

        self.load_data_file()
        samples = self.count
        start = 0

        if self.subsample:
            samples = self.subsample if samples > self.subsample else samples
        elif self.validation:
            start = samples // 2

        for i in range(int(start), int(samples)):
            c_input = input_folder / "{:02d}".format(subfolder)
            c_input = c_input / "{:06d}.npz".format(i)
            x_int, x_cat, y = self.get_data(i)

            if self.save_preprocessed_features:
                if not c_input.parent.exists():
                    c_input.parent.mkdir(parents=True)

                sample = {
                    self.dense_features: np.log1p(x_int),
                    self.lso_features: np.dot(np.expand_dims(np.linspace(0, self.batch - 1, num=self.batch), -1),
                                              np.ones((1, self.cat_feat))).T
                }

                for name in self.sparse_features.keys():
                    x_cat_batch = x_cat[:, self.sparse_features[name]]
                    x_cat_batch = x_cat_batch % self.max_ind_range if self.max_ind_range is not None else x_cat_batch
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
                label = y if not np.isscalar(y) else [y, ]
                annotations.append(ClassificationAnnotation(identifiers, label))
            else:
                for j in range(i, i + self.batch):
                    identifiers = [
                        "{}_{}{}{}".format(self.dense_features, j, self.separator, c_file),
                        "{}_{}{}{}".format(self.lso_features, j, self.separator, c_file),
                    ]
                    for name in self.sparse_features.keys():
                        identifiers.append("{}_{}{}{}".format(name, j, self.separator, c_file))
                    label = y[j, ...] if not np.isscalar(y[j, ...]) else [y[j, ...], ]
                    annotations.append(ClassificationAnnotation(identifiers, label))

        self.close_data_file()
        return ConverterReturn(annotations, None, None)
