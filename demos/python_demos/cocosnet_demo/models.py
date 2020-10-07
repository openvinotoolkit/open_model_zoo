"""
 Copyright (C) 2020 Intel Corporation
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
from openvino.inference_engine import IECore


class CorrespondenceModel:
    def __init__(self, ie_core, model_xml, model_bin, device='CPU'):
        print("Reading IR for correspondence model ....")
        self.network = ie_core.read_network(model_xml, model_bin)
        self.inputs = list(self.network.input_info.keys())
        self.outputs = list(self.network.outputs.keys())
        assert len(self.inputs) == 3, "Three inputs are expected"
        assert len(self.outputs) == 2, "Two outputs are expected"
        self.input_semantics, self.reference_image, self.reference_semantics = self.inputs
        self.warp_out, self.warp_mask = self.outputs

        print("Loading Correspondence IR to the plugin...")
        self.exec_net = ie_core.load_network(network=self.network, device_name=device)

    def infer(self, input_semantics, reference_image, reference_semantics):
        input_data = {
            self.input_semantics: input_semantics,
            self.reference_image: reference_image,
            self.reference_semantics: reference_semantics
        }
        result = self.exec_net.infer(input_data)
        return result


class SingleInputModel:
    def __init__(self, ie_core, model_xml, model_bin, device='CPU'):
        print("Reading IR for {} model ....".format(self.model_type))
        self.network = ie_core.read_network(model_xml, model_bin)
        assert len(self.network.input_info) == 1, "One input is expected"
        assert len(self.network.outputs) == 1, "One input is expected"
        self.input_name = next(iter(self.network.input_info))
        self.output_name = next(iter(self.network.outputs))

        print("Loading IR to the plugin...")
        self.exec_net = ie_core.load_network(network=self.network, device_name=device)

    def infer(self, input):
        input_data = {self.input_name: input}
        result = self.exec_net.infer(input_data)
        return result[self.output_name]


class GenerativeModel(SingleInputModel):
    def __init__(self, ie_core, model_xml, model_bin, device='CPU'):
        self.model_type = "generative"
        super().__init__(ie_core, model_xml, model_bin, device='CPU')


class CocosnetModel:
    def __init__(self, corr, gen):
        self.correspondence = corr
        self.generator = gen

    def infer(self, inputs={}):
        if inputs:
            corr_out = self.correspondence.infer(**inputs)
            gen_input = np.concatenate((corr_out[self.correspondence.warp_out],
                                        inputs['input_semantics']), axis=1)
            return self.generator.infer(gen_input)


class SegmentationModel(SingleInputModel):
    def __init__(self, ie_core, model_xml, model_bin, device='CPU'):
        self.model_type = "semantic_segmentation"
        super().__init__(ie_core, model_xml, model_bin, device='CPU')
