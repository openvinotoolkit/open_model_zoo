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
import cv2
import numpy as np
from .base_representation import BaseRepresentation
TAG_FLOAT = 202021.25


class OpticalFlowRepresentation(BaseRepresentation):
    pass


class OpticalFlowAnnotation(OpticalFlowRepresentation):
    def __init__(self, identifier, path_to_gt):
        super().__init__(identifier)
        self.path_to_gt = path_to_gt
        self._value = None

    @property
    def value(self):
        return self._value if self._value is not None else self._load_flow()

    @value.setter
    def value(self, flow):
        self._value = flow

    def _load_flow(self):
        if self._value is not None:
            return self._value
        data_source = self.metadata.get('segmentation_masks_source') or self.metadata.get('additional_data_source')
        if data_source is None:
            data_source = self.metadata['data_source']
        src_file = data_source / self.path_to_gt
        if self.path_to_gt.lower().endswith('.flo'):
            with open(str(src_file), 'rb') as f:
                tag = float(np.fromfile(f, np.float32, count=1)[0])
                assert tag == TAG_FLOAT
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                flow = np.fromfile(f, np.float32, count=h * w * 2)
                flow.resize((h, w, 2))
            return flow

        if self.path_to_gt.lower().endswith('.png'):
            flow_raw = cv2.imread(str(src_file), -1)
            flow = flow_raw[:, :, 2:0:-1].astype(np.float32)
            flow = flow - 32768
            flow = flow / 64

            flow[np.abs(flow) < 1e-10] = 1e-10

            invalid = (flow_raw[:, :, 0] == 0)
            flow[invalid, :] = 0
            return flow

        if self.path_to_gt.lower().endswith('.pfm'):
            with open(str(src_file), 'rb') as f:
                tag = f.readline().rstrip().decode("utf-8")
                assert tag == 'PF'
                dims = f.readline().rstrip().decode("utf-8")
                w, h = map(int, dims.split(' '))
                scale = float(f.readline().rstrip().decode("utf-8"))

                flow = np.fromfile(f, '<f') if scale < 0 else np.fromfile(f, '>f')
                flow = np.reshape(flow, (h, w, 3))[:, :, 0:2]
                flow = np.flipud(flow)
            return flow
        raise ValueError('Unsupported flow format {}'.format(self.path_to_gt))


class OpticalFlowPrediction(OpticalFlowRepresentation):
    def __init__(self, identifier, flow):
        super().__init__(identifier)
        self.value = flow
