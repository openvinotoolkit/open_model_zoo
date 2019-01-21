"""
Copyright (c) 2018 Intel Corporation

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
from PIL import Image
import numpy as np
from ..utils import get_path
from ..dependency import ClassProvider


class BaseReader(ClassProvider):
    __provider_type__ = 'reader'

    def __call__(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    def read(self, data):
        raise NotImplementedError


class OpenCVImageReader(BaseReader):
    __provider__ = 'opencv_imread'

    def read(self, data):
        return cv2.imread(str(get_path(data)))


class PillowImageReader(BaseReader):
    __provider__ = 'pillow_imread'

    def read(self, data):
        return np.array(Image.open(get_path(data)))[:, :, ::-1]


class OpenCVFrameReader(BaseReader):
    __provider__ = 'opencv_capture'

    def __init__(self):
        self.source = None

    def read(self, data):
        frame_id = int(data.parts[-1])
        source = str(data.parent)

        # source video changed, capture initialization
        if source != self.source:
            self.source = source
            self.videocap = cv2.VideoCapture(self.source)

        # set frame position for reading
        self.videocap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = self.videocap.read()
        if not success:
            raise EOFError('frame with {} index does not exists in {}'.format(frame_id, self.source))

        return frame
