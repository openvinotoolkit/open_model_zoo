"""
 Copyright (c) 2021 Intel Corporation

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
from .utils import softmax

from .image_model import ImageModel


class Classification(ImageModel):
    def __init__(self, model_adapter, topk = 1, labels = None, resize_type='crop'):
        super().__init__(model_adapter, resize_type=resize_type)
        self._check_io_number(1, 1)
        self.topk = topk
        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = self._load_labels(labels) if labels else None
        self.apply_softmax = "Softmax" not in [layer_info.type for _, layer_info in self.model_adapter._get_meta_from_ngraph().items()]
        self.out_layer_name = self._get_outputs()

    @staticmethod
    def _load_labels(labels_file):
        with open(labels_file, 'r') as f:
            labels = []
            for s in f:
                begin_idx = s.find(' ')
                if (begin_idx == -1):
                    raise RuntimeError('The labels file has incorrect format.')
                end_idx = s.find(',')
                labels.append(s[(begin_idx + 1):end_idx])
        return labels

    def _get_outputs(self):
        layer_name = next(iter(self.outputs))
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) != 2 and len(layer_shape) != 4:
            raise RuntimeError('The Classification model wrapper supports topologies only with 2D or 4D output')
        if len(layer_shape) == 4 and (layer_shape[2] != 1 or layer_shape[3] != 1):
            raise RuntimeError('The Classification model wrapper supports topologies only with 4D '
                               'output which has last two dimensions of size 1')
        if self.labels:
            if (layer_shape[1] == len(self.labels) + 1):
                self.labels.insert(0, 'other')
                self.logger.warning("\tInserted 'other' label as first.")
            if layer_shape[1] != len(self.labels):
                raise RuntimeError("Model's number of classes and parsed "
                                'labels must match ({} != {})'.format(layer_shape[1], len(self.labels)))
        return layer_name

    def postprocess(self, outputs, meta):
        outputs = outputs[self.out_layer_name].squeeze()
        indices = np.argpartition(outputs, -self.topk)[-self.topk:]
        scores = outputs[indices]

        desc_order = scores.argsort()[::-1]
        scores = scores[desc_order]
        indices = indices[desc_order]
        if self.apply_softmax:
            scores = softmax(scores)
        labels = [self.labels[i] if self.labels else "" for i in indices]
        return list(zip(indices, labels, scores))