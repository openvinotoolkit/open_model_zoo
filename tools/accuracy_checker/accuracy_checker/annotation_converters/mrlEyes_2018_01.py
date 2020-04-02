"""
Copyright (c) 2019 Intel Corporation

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

import re
from pathlib import Path

from ..config import PathField
from ..representation import ClassificationAnnotation
from ..utils import get_path, read_txt

from .format_converter import BaseFormatConverter, ConverterReturn


class mrlEyes_2018_01_Converter(BaseFormatConverter):
    """
    mrlEyes_2018 dataset converter. 
    """

    __provider__ = 'mrlEyes_2018_01'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'data_dir': PathField(is_directory=True, description="Path to mrlEyes_2018_01 dataset root directory.")
        })

        return configuration_parameters

    def configure(self):
        """
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        """
        self.data_dir = self.config['data_dir']
        self.use_every = self.config['use_every']

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically got from command line arguments or config file in method configure

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata (if provided)
        """

        dataset_directory = Path(self.data_dir)

        annotations = []
        for i, file in enumerate(dataset_directory.rglob('*.png')):
                subj_id, img_num, gender, glasses, eye_state, reflection, light_cond, sensor_type = file.stem.split("_")
                if i % 10 == 0 or self.use_every:
                    annotations.append(ClassificationAnnotation(full_path, int(eye_state)))

        annotations = self._convert_annotations(images_dir, labels, progress_callback, progress_interval)

        # convert label list to label map
        label_map = {0:'closed', 1: 'open'} 
        metadata = {'label_map': label_map}

        return ConverterReturn(annotations, metadata, None)
