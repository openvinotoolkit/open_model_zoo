"""
 Copyright (c) 2022 Intel Corporation

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

import cv2
import numpy as np

from .image_model import ImageModel
from .model import WrapperError


class VideoBackgroundMatting(ImageModel):
    __model__ = 'Robust-video-matting'

    def __init__(self, model_adapter, configuration, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self._check_io_number((5, ), (6, ))
        self.output_blob_name = self._get_outputs()
        self.rec_map = self.get_inputs_map()
        self.rec = self.initialize_rec()

    @classmethod
    def parameters(cls):
        return super().parameters()

    def _get_inputs(self):
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 4 and metadata.shape[1] == 3:
                image_blob_names.append(name)
        if not image_blob_names:
            raise WrapperError(self.__model__, 'Compatible inputs are not found')
        return image_blob_names, image_info_blob_names

    def _get_outputs(self):
        image_blob_names = {}
        for name, metadata in self.outputs.items():
            if len(metadata.shape) == 4 and metadata.shape[1] == 3:
                image_blob_names['fgr'] = name
            elif len(metadata.shape) == 4 and metadata.shape[1] == 1:
                image_blob_names['pha'] = name
        if len(image_blob_names) != 2:
            raise WrapperError(self.__model__, 'Compatible outputs are not found')
        return image_blob_names

    def get_inputs_map(self):
        rec_map = {}
        for in_name, in_meta in self.inputs.items():
            if in_meta.shape[1] not in [1, 3]:
                for out_name, out_meta in self.outputs.items():
                    if in_meta.shape == out_meta.shape:
                        rec_map[in_name] = out_name
                        break
        return rec_map

    def preprocess(self, inputs):
        dict_inputs, meta = super().preprocess(inputs)
        dict_inputs.update(self.rec)
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        fgr = outputs[self.output_blob_name['fgr']]
        pha = outputs[self.output_blob_name['pha']]
        self.rec = {in_name: outputs[out_name] for in_name, out_name in self.rec_map.items()}
        fgr = fgr[0].transpose(1, 2, 0)
        pha = pha[0].transpose(1, 2, 0)
        h, w = meta['original_shape'][:2]
        fgr = cv2.cvtColor(cv2.resize(fgr, (w, h)), cv2.COLOR_RGB2BGR)
        pha = np.expand_dims(cv2.resize(pha, (w, h)), axis=-1)
        return fgr, pha

    def initialize_rec(self):
        rec = {}
        for name, metadata in self.inputs.items():
            if name in self.rec_map.keys():
                rec[name] = np.zeros(metadata.shape, dtype=np.float32)
        return rec


class ImageMattingWithBackground(ImageModel):
    __model__ = 'Background-matting'

    def __init__(self, model_adapter, configuration, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self._check_io_number((2, ), (2, 3))
        self.output_blob_name = self._get_outputs()
        self.n, self.c, self.h, self.w = self.set_input_shape()

    @classmethod
    def parameters(cls):
        return super().parameters()

    def _get_inputs(self):
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 4 and metadata.shape[1] == 3:
                image_blob_names.append(name)
        if len(image_blob_names) != 2:
            raise WrapperError(self.__model__, 'Compatible inputs are not found')
        return image_blob_names, image_info_blob_names

    def set_input_shape(self):
        shapes = [tuple(self.inputs[name].shape) for name in self.image_blob_names]
        if len(set(shapes)) != 1:
            raise WrapperError(self.__model__, 'Image inputs have incompatible shapes: {}'.format(shapes))
        return shapes[0]

    def _get_outputs(self):
        image_blob_names = {}
        for name, metadata in self.outputs.items():
            if len(metadata.shape) == 4 and metadata.shape[1] == 3:
                image_blob_names['fgr'] = name
            elif len(metadata.shape) == 4 and metadata.shape[1] == 1:
                image_blob_names['pha'] = name
        if len(image_blob_names) != 2:
            raise WrapperError(self.__model__, 'Compatible outputs are not found')
        return image_blob_names

    def preprocess(self, inputs):
        dict_inputs = {}
        target_shape = None
        for name, image in inputs.items():
            self.image_blob_name = name
            dict_input, meta = super().preprocess(image)
            dict_inputs.update(dict_input)
            if target_shape is None:
                target_shape = meta['original_shape']
            elif meta['original_shape'] != target_shape:
                raise WrapperError(self.__model__, 'Image inputs must have equal shapes but got: {} vs {}'
                                    .format(target_shape, meta['original_shape']))
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        fgr = outputs[self.output_blob_name['fgr']]
        pha = outputs[self.output_blob_name['pha']]
        fgr = fgr[0].transpose(1, 2, 0)
        pha = pha[0].transpose(1, 2, 0)
        h, w = meta['original_shape'][:2]
        fgr = cv2.cvtColor(cv2.resize(fgr, (w, h)), cv2.COLOR_RGB2BGR)
        pha = np.expand_dims(cv2.resize(pha, (w, h)), axis=-1)
        return fgr, pha
