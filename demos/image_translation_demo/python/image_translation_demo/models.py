"""
 Copyright (C) 2020-2022 Intel Corporation
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
    def __init__(self, core, model_xml, device='CPU'):
        model = core.read_model(model_xml)
        if len(model.inputs) != 3:
            raise RuntimeError("The CocosnetModel expects 3 input layers")
        if len(model.outputs) != 1:
            raise RuntimeError("The CocosnetModel expects 1 output layer")

        inputs = [node.get_any_name() for node in model.inputs]
        self.input_semantics, self.reference_image, self.reference_semantics = inputs
        self.input_semantic_size = model.input(self.input_semantics).shape
        self.input_image_size = model.input(self.reference_image).shape

        compiled_model = core.compile_model(model, device)
        self.infer_request = compiled_model.create_infer_request()

    def infer(self, input_semantics, reference_image, reference_semantics):
        input_data = {
            self.input_semantics: input_semantics,
            self.reference_image: reference_image,
            self.reference_semantics: reference_semantics
        }
        result = self.infer_request.infer(input_data)
        return next(iter(result.values()))


class SegmentationModel:
    def __init__(self, core, model_xml, device='CPU'):
        model = core.read_model(model_xml)
        if len(model.inputs) != 1:
            raise RuntimeError("The SegmentationModel expects 1 input layer")
        if len(model.outputs) != 1:
            raise RuntimeError("The SegmentationModel expects 1 output layer")

        self.input_tensor_name = model.inputs[0].get_any_name()
        self.input_size = model.inputs[0].shape

        compiled_model = core.compile_model(model, device)
        self.infer_request = compiled_model.create_infer_request()

    def infer(self, input):
        input_data = {self.input_tensor_name: input}
        result = self.infer_request.infer(input_data)
        return next(iter(result.values()))
