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

import numpy as np

from ..config import BoolField, StringField, NumberField, ConfigError
from .data_reader import BaseReader
from ..utils import UnsupportedPackage, get_path

try:
    import nibabel as nib
except ImportError as import_error:
    nib = UnsupportedPackage("nibabel", import_error.msg)

try:
    import pydicom
except ImportError as import_error:
    pydicom = UnsupportedPackage("pydicom", import_error.msg)


class NiftiImageReader(BaseReader):
    __provider__ = 'nifti_reader'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'channels_first': BoolField(optional=True, default=False,
                                        description='Allows read files and transpose in order where channels first.'),
            'frame_separator': StringField(optional=True, default='#',
                                           description="Separator between filename and frame number"),
            'multi_frame': BoolField(optional=True, default=False,
                                     description="Add annotation for each frame in source file"),
            'to_4D': BoolField(optional=True, default=True, description="Ensure that data are 4D"),
            'frame_axis': NumberField(optional=True, default=-1, description="Frames dimension axis"),
        })
        return parameters

    def configure(self):
        if isinstance(nib, UnsupportedPackage):
            nib.raise_error(self.__provider__)
        self.channels_first = self.get_value_from_config('channels_first')
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.frame_axis = int(self.get_value_from_config('frame_axis'))
        self.frame_separator = self.get_value_from_config('frame_separator')
        self.multi_frame = self.get_value_from_config('multi_frame')
        self.to_4D = self.get_value_from_config('to_4D')
        self.data_layout = self.get_value_from_config('data_layout')

        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=True)

    def read(self, data_id):
        if self.multi_frame:
            parts = data_id.split(self.frame_separator)
            frame_number = int(parts[1])
            data_id = parts[0]
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        nib_image = nib.load(str(get_path(data_path)))
        image = np.array(nib_image.dataobj)
        if self.multi_frame:
            image = image[:, :, frame_number]
            image = np.expand_dims(image, 0)
        if self.to_4D:
            if len(image.shape) != 4:  # Make sure 4D
                image = np.expand_dims(image, -1)
            image = np.transpose(image, (3, 0, 1, 2) if self.channels_first else (2, 1, 0, 3))

        return image


class DicomReader(BaseReader):
    __provider__ = 'dicom_reader'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config)
        if isinstance(pydicom, UnsupportedPackage):
            pydicom.raise_error(self.__provider__)

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        dataset = pydicom.dcmread(str(data_path))
        return dataset.pixel_array
