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

from ..representation import OAR3DTilingSegmentationAnnotation
from ..config import NumberField, StringField, PathField
from .format_converter import DirectoryBasedAnnotationConverter
from .format_converter import ConverterReturn


class OAR3DTilingConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'ge_tiling'
    annotation_types = (OAR3DTilingSegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            "depth": NumberField(optional=True, default=32, description="Tile depth."),
            "height": NumberField(optional=True, default=128, description="Tile height."),
            "width": NumberField(optional=True, default=128, description="Tile width."),
            "input": StringField(optional=True, default="inputs", description="Name of input data variable."),
            "output": StringField(optional=True, default="outputs", description="Name of output data variable."),
            "preprocessed_dir": PathField(optional=False, is_directory=True, check_exists=True,
                                          description="Preprocessed dataset location")
        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.wD = self.get_value_from_config('depth')
        self.wW = self.get_value_from_config('width')
        self.wH = self.get_value_from_config('height')
        self.input = self.get_value_from_config('input')
        self.output = self.get_value_from_config('output')
        self.preprocessed_dir = self.get_value_from_config('preprocessed_dir')

    def convert(self, check_content=False, **kwargs):

        data_folder = Path(self.data_dir)
        preprocessed_folder = Path(self.preprocessed_dir)
        input_folder = preprocessed_folder / 'input'
        mask_folder = preprocessed_folder / 'mask'

        for folder in [input_folder, mask_folder]:
            if not folder.exists():
                folder.mkdir()

        annotations = []
        for src in data_folder.glob('*.npz'):

            src_input_folder = input_folder / "{}_{}_{}_{}".format(src.stem, self.wD, self.wH, self.wW)
            src_mask_folder = mask_folder / "{}_{}_{}_{}".format(src.stem, self.wD, self.wH, self.wW)

            for folder in [src_input_folder, src_mask_folder]:
                if not folder.exists():
                    folder.mkdir()

            data = np.load(src)
            inputs = data[self.input]
            outputs = data[self.output]
            B, _, D, H, W = inputs.shape
            B, _, CLS = outputs.shape
            outputs = outputs.reshape([B, D, H, W, CLS])

            D = int(D / self.wD) * self.wD
            H = int(H / self.wH) * self.wH
            W = int(W / self.wW) * self.wW

            for cD in range(0, D, self.wD):
                for cH in range(0, H, self.wH):
                    for cW in range(0, W, self.wW):

                        input_name, mask_name = self.preprocess(src_mask_folder, src_input_folder, cD, cH, cW,
                                                                inputs, outputs, CLS)

                        annotations.append(OAR3DTilingSegmentationAnnotation(
                            str(input_name.relative_to(preprocessed_folder)),
                            str(mask_name.relative_to(preprocessed_folder))
                        ))

        return ConverterReturn(annotations, None, None)

    def preprocess(self, src_mask_folder, src_input_folder, cD, cH, cW, inputs, outputs, CLS):
        mask_name = src_mask_folder / "{}_{}_{}.npy".format(cD, cH, cW)
        input_name = src_input_folder / "{}_{}_{}.npy".format(cD, cH, cW)

        if not (mask_name.exists() and input_name.exists()):

            inp = np.zeros([1, self.wD, self.wH, self.wW], dtype=float)
            ref = np.zeros([self.wD, self.wH, self.wW, CLS], dtype=float)
            for d in range(self.wD):
                for h in range(self.wH):
                    for w in range(self.wW):
                        inp[0, d, h, w] = inputs[0, 0, cD + d, cH + h, cW + w]
                        for c in range(CLS):
                            ref[d, h, w, c] = outputs[0, cD + d, cH + h, cW + w, c]

            ref = ref.reshape([self.wD * self.wW * self.wH, CLS])

            np.save(input_name, inp)
            np.save(mask_name, ref)

        return input_name, mask_name
