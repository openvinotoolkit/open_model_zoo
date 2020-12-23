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

class CocosnetModel:
    def __init__(self, ie_core, model_xml, model_bin, device='CPU'):
        print("Reading IR for CoCosNet model ....")
        self.network = ie_core.read_network(model_xml, model_bin)
        assert len(self.network.input_info) == 3, "Three inputs are expected"
        assert len(self.network.outputs) == 1, "One output is expected"
        self.inputs = list(self.network.input_info.keys())
        self.output_name = next(iter(self.network.outputs.keys()))
        self.input_semantics, self.reference_image, self.reference_semantics = self.inputs

        print("Loading CoCosNet IR to the plugin...")
        self.exec_net = ie_core.load_network(network=self.network, device_name=device)
        self.input_semantic_size = self.network.input_info[self.input_semantics].input_data.shape
        self.input_image_size = self.network.input_info[self.reference_image].input_data.shape

    def infer(self, input_semantics, reference_image, reference_semantics):
        input_data = {
            self.input_semantics: input_semantics,
            self.reference_image: reference_image,
            self.reference_semantics: reference_semantics
        }
        result = self.exec_net.infer(input_data)
        return result[self.output_name]


class SegmentationModel:
    def __init__(self, ie_core, model_xml, model_bin, device='CPU'):
        print("Reading IR for segmentation model ....")
        self.network = ie_core.read_network(model_xml, model_bin)
        assert len(self.network.input_info) == 1, "One input is expected"
        assert len(self.network.outputs) == 1, "One output is expected"
        self.input_name = next(iter(self.network.input_info))
        self.output_name = next(iter(self.network.outputs))

        print("Loading IR to the plugin...")
        self.exec_net = ie_core.load_network(network=self.network, device_name=device)
        self.input_size = self.network.input_info[self.input_name].input_data.shape

    def infer(self, input):
        input_data = {self.input_name: input}
        result = self.exec_net.infer(input_data)
        return result[self.output_name]
