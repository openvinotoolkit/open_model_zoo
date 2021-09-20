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

from pathlib import Path
import numpy as np

from ..representation import ImageProcessingAnnotation
from ..utils import get_path
from ..config import StringField, PathField, BoolField
from .format_converter import DirectoryBasedAnnotationConverter
from .format_converter import ConverterReturn
from ..representation.image_processing import GTLoader

class KSpaceMRIConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'k_space_mri'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'image_folder': StringField(optional=True, default='images',
                                        description="K-space images folder."),
            'reconstructed_folder': StringField(optional=True, default='reconstructed',
                                                description="Reconstructed images folder."),
            'sampled_folder': StringField(optional=True, default='sampled', description="Sampled k-space data folder."),
            'mask_file': PathField(optional=False, default=None, description="K-space mask filename"),
            'stats_file': PathField(optional=False, default=None, description="K-space normalization filename"),
            'skip_dumps': BoolField(optional=True, default=True,
                                    description="Skips dumps of reconstructed and sampled data"),

        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.image_folder = self.get_value_from_config('image_folder')
        self.reconstructed_folder = self.get_value_from_config('reconstructed_folder')
        self.sampled_folder = self.get_value_from_config('sampled_folder')
        self.mask_file = self.get_value_from_config('mask_file')
        self.stats_file = self.get_value_from_config('stats_file')
        self.skip_dumps = self.get_value_from_config('skip_dumps')

    def convert(self, check_content=False, **kwargs):
        image_folder = Path(self.image_folder)
        sampled_folder = Path(self.sampled_folder)
        reconstructed_folder = Path(self.reconstructed_folder)
        image_dir = get_path(self.data_dir / image_folder, is_directory=True)
        if not self.skip_dumps and not (self.data_dir / reconstructed_folder).exists():
            (self.data_dir / reconstructed_folder).mkdir(parents=True, exist_ok=True)
        reconstructed_dir = get_path(self.data_dir / reconstructed_folder, is_directory=True)
        if not self.skip_dumps and not (self.data_dir / sampled_folder).exists():
            (self.data_dir / sampled_folder).mkdir(parents=True, exist_ok=True)
        sampled_dir = get_path(self.data_dir / sampled_folder, is_directory=True)

        annotations = []
        var_sampling_mask = np.load(self.mask_file)
        stats = np.load(self.stats_file)
        frame_separator = '#'

        for file_in_dir in image_dir.iterdir():
            data = np.load(file_in_dir)
            total_frames, width, height, _ = data.shape
            norm = np.sqrt(width * height)
            for frame_cnt in range(total_frames):
                if not self.skip_dumps:
                    kspace_data = data[frame_cnt, ...] / norm
                    rec_data = np.abs(np.fft.ifft2(kspace_data[:, :, 0] + 1j * kspace_data[:, :, 1]))
                    kspace_data[var_sampling_mask, :] = 0
                    kspace_data = (kspace_data - stats[0]) / stats[1]
                reconstructed_name = "{}{}{}{}".format(str(reconstructed_dir / file_in_dir.stem),
                                                       frame_separator, frame_cnt, file_in_dir.suffix)
                sampled_name = "{}{}{}{}".format(str(sampled_dir / file_in_dir.stem), frame_separator,
                                                 frame_cnt, file_in_dir.suffix)
                if not self.skip_dumps:
                    np.save(reconstructed_name, rec_data)
                    np.save(sampled_name, kspace_data)
                annotation = ImageProcessingAnnotation(str(Path(sampled_name).relative_to(self.data_dir)),
                                                       str(Path(reconstructed_name).relative_to(self.data_dir)),
                                                       gt_loader=GTLoader.NUMPY)
                annotations.append(annotation)

        return ConverterReturn(annotations, None, None)
