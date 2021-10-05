"""
 Copyright (c) 2020 Intel Corporation

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

import ngraph as ng
from openvino.inference_engine import IENetwork

from .image_model import ImageModel


class Classification(ImageModel):
    def __init__(self, ie, model_path, labels, ntop, resize_type='crop'):
        super().__init__(ie, model_path, resize_type=resize_type)
        self._check_io_number(1, 1)
        if self.h != self.w:
            raise RuntimeError('Model input has incorrect image shape. Must be NxN square. '
                                'Got {} x {}.'.format(self.h, self.w))

        self.ntop = ntop
        self.labels = self._load_labels(labels)
        self._check_outputs()

        function = ng.function_from_cnn(self.net)
        if 'Softmax' not in [node.get_type_name() for node in function.get_ops()]:
            logits = function.get_output_op(0)
            logits = logits.inputs()[0].get_source_output().get_node()
            logits.set_friendly_name('logits')
            probs_node = ng.softmax(logits, 1)
        else:
            probs_node = function.get_output_op(0)
            probs_node = probs_node.inputs()[0].get_source_output().get_node()
            probs_node.set_friendly_name('probs')

        topk_node = ng.topk(probs_node, ntop, 1, "max", "value")
        f = ng.Function(
            [ng.result(topk_node.outputs()[0], name='TopK_scores'),
             ng.result(topk_node.outputs()[1], name='TopK_indices')],
            function.get_parameters(), 'classification_res')
        self.net = IENetwork(ng.impl.Function.to_capsule(f))
        self.out_blob_names = list(self.net.outputs.keys())
        self.out_blob_names.sort() # sort names to make sure that out_blob_names[0] - scores, out_blob_names[1] - indices

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

    def _check_outputs(self):
        out_blob_name = next(iter(self.net.outputs))
        out_blob = self.net.outputs[out_blob_name]
        out_size = out_blob.shape
        if len(out_size) != 2 and len(out_size) != 4:
            raise RuntimeError('The Classification model wrapper supports topologies only with 2D or 4D output')
        if len(out_size) == 4 and (out_size[2] != 1 or out_size[3] != 1):
            raise RuntimeError('The Classification model wrapper supports topologies only with 4D '
                               'output which has last two dimensions of size 1')
        if (out_size[1] == len(self.labels) + 1):
            self.labels.insert(0, 'other')
            self.logger.warning("\tInserted 'other' label as first.")
        if out_size[1] != len(self.labels):
            raise RuntimeError("Model's number of classes and parsed "
                              'labels must match ({} != {})'.format(out_size[1], len(self.labels)))

    def postprocess(self, outputs, meta):
        scores = outputs[self.out_blob_names[0]].squeeze()
        indices = outputs[self.out_blob_names[1]].squeeze()
        return list(zip(indices, scores))
