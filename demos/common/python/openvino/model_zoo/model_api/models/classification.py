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

import numpy as np

from .image_model import ImageModel


class Classification(ImageModel):
    def __init__(self, ie, model_path, labels, ntop, resize_type='crop', logger=None):
        super().__init__(ie, model_path, resize_type=resize_type)
        self._check_io_number(1, 1)
        self._check_inputs()
        self.ntop = ntop
        self.log = logger
        self.labels = self._load_labels(labels)
        self.output_blob_name = self._get_outputs()

    @staticmethod
    def _load_labels(labels_file):
        with open(labels_file, 'r') as f:
            labels = []
            for s in f:
                begin_idx = s.find(' ')
                if (begin_idx == -1):
                    raise Exception("The labels file has incorrect format.")
                end_idx = s.find(',')
                labels.append(s[(begin_idx + 1):end_idx])
        return labels

    def _check_inputs(self):
        in_size = self.inputs[self.image_blob_name].input_data.shape
        if len(in_size) == 4 and in_size[1] != 3:
            raise Exception("3-channel 4-dimensional model's input is expected")
        if in_size[2] != in_size[3]:
            raise Exception("Model input has incorrect image shape. Must be NxN square."
                                " Got " + str(in_size[2]) + "x" + str(in_size[3]) + ".")

    def _get_outputs(self):
        out_blob_name = next(iter(self.net.outputs))
        out_blob = self.net.outputs[out_blob_name]
        out_size = out_blob.shape
        if len(out_size) != 2 and len(out_size) != 4:
            raise Exception("Demo supports topologies only with 2-dimensional or 4-dimensional output")
        if len(out_size) == 4 and out_size[2] != 1 and out_size[3] != 1:
            raise Exception("Demo supports topologies only with 4-dimensional output which has last two dimensions of size 1")

        if (out_size[1] == len(self.labels) + 1):
            self.labels.insert(0, "other")
            if self.log:
                self.log.warning("\tInserted 'other' label as first.")
        if out_size[1] != len(self.labels):
            raise Exception("Model's number of classes and parsed labels must match \
                 (" + str(out_size[1]) + " and " + str(len(self.labels)) + ')')

        return out_blob_name

    def postprocess(self, outputs, meta):
        probs = outputs[self.output_blob_name].squeeze()
        indices = np.argsort(probs)
        max_indices = indices[-self.ntop:][::-1] # indices sorted by probs in descended order
        result = [(i, probs[i]) for i in max_indices]
        return result
