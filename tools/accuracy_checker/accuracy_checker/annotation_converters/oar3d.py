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

from pathlib import Path
import warnings
import numpy as np

from ..representation import BrainTumorSegmentationAnnotation, OAR3DTilingSegmentationAnnotation
from ..config import StringField, PathField, BoolField, NumberField
from .format_converter import DirectoryBasedAnnotationConverter
from ..representation.segmentation_representation import GTMaskLoader
from .format_converter import ConverterReturn


class OAR3DTilingConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'ge_tiling'
    annotation_types = (BrainTumorSegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            "depth": NumberField(optional=True, default=32, description="Tile depth."),
            "height": NumberField(optional=True, default=128, description="Tile height."),
            "width": NumberField(optional=True, default=128, description="Tile width."),
            "batch": NumberField(optional=True, default=5, description="Number of tiles in batch."),
        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.wD = self.get_value_from_config('depth')
        self.wW = self.get_value_from_config('width')
        self.wH = self.get_value_from_config('height')
        self.batch = self.get_value_from_config('batch')

    def convert(self, check_content=False, **kwargs):
        data_folder = Path(self.data_dir)

        annotations = []
        for src in data_folder.glob('*.npz'):
            data = np.load(src)
            N, C, D, H, W = data['inputs'].shape

            D = int(D / self.wD) * self.wD
            H = int(H / self.wH) * self.wH
            W = int(W / self.wW) * self.wW

            tiles = []
            for cD in range(0, D, self.wD):
                for cH in range(0, H, self.wH):
                    for cW in range(0, W, self.wW):
                        tiles.append([cD, cH, cW])
                        if len(tiles) == self.batch:
                            for tile in tiles:
                                tD,tH,tW = tile
                                annotations.append(OAR3DTilingSegmentationAnnotation(
                                    str(data_folder / src),
                                    str(data_folder / src),
                                    # tiles,
                                    tD,
                                    tH,
                                    tW,
                                    self.wD,
                                    self.wH,
                                    self.wW))
                            tiles = []

        return ConverterReturn(annotations, None, None)
