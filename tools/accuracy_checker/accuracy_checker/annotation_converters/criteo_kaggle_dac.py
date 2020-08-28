"""
Copyright (c) 2018-2020 Intel Corporation

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
from ..config import NumberField, StringField, PathField, BoolField
from .format_converter import BaseFormatConverter
from .format_converter import ConverterReturn

class CriteoKaggleDACConverter(BaseFormatConverter):

    __provider__ = 'criteo_kaggle_dac'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            "batch": NumberField(optional=True, default=128, description="Model batch"),
            "subsample_size": NumberField(optional=True, default=0,
                                          description="Limit total record count to batch * subsample size"),
            "validation": BoolField(optional=True, default=True,
                                    description="Allows to use half of dataset for validation purposes"),
            "separator": StringField(optional=True, default='#',
                                     description="Separator between input identifier and file identifier"),
            "preprocessed_dir": PathField(optional=False, is_directory=True, check_exists=True,
                                          description="Preprocessed dataset location")
        })

        return parameters

    def configure(self):
        self.src = self.get_value_from_config('testing_file')
        self.batch = self.get_value_from_config('batch')
        self.subsample = self.get_value_from_config('subsample_size')
        self.validation = self.get_value_from_config('validation')
        self.separator = self.get_value_from_config('separator')
        self.preprocessed_dir = self.get_value_from_config('preprocessed_dir')

    def convert(self, check_content=False, **kwargs):

        preprocessed_folder = Path(self.preprocessed_dir)
        input_folder = preprocessed_folder / "bs{}".format(self.batch) / 'input'

        if not input_folder.exists():
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

            if not c_input.exists():
                c_input.mkdir(parents=True)

            c_input = c_input / "{:06d}.npz".format(i)

            sample = {
                "input.1": np.log1p(x_int[i:i+self.batch, ...]),
                "lS_i": x_cat[i:i+self.batch, ...],
                "lS_o": np.dot(np.expand_dims(np.linspace(0, self.batch - 1, num=self.batch), -1),
                               np.ones((1, cat_feat)))
            }

            np.savez_compressed(str(c_input), **sample)

            filecnt += 1
            filecnt %= 0x100

            subfolder = subfolder + 1 if filecnt == 0 else subfolder

            c_file = str(c_input.relative_to(preprocessed_folder))

            for j in range(i, i + self.batch):
                annotations.append(ClassificationAnnotation(
                    [
                        "input.1_{}{}{}".format(j, self.separator, c_file),
                        "lS_i_{}{}{}".format(j, self.separator, c_file),
                        "lS_o_{}{}{}".format(j, self.separator, c_file),
                    ],
                    y[j, ...]
                ))

        return ConverterReturn(annotations, None, None)
