"""
 Copyright (c) 2019-2022 Intel Corporation

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

import logging as log
from openvino.runtime import Core, get_version


def load_core(device, cpu_extension=None):
    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()
    if device == "CPU" and cpu_extension:
        core.add_extension(cpu_extension, "CPU")

    return core


class IEModel:  # pylint: disable=too-few-public-methods
    """ Class that allows worknig with Inference Engine model. """

    def __init__(self, model_path, device, core, num_requests, model_type, output_shape=None):
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)

        if len(self.model.inputs) != 1:
            raise RuntimeError("The {} wrapper supports only models with 1 input layer".format(model_type))

        compiled_model = core.compile_model(self.model, device)
        self.infer_requests = [compiled_model.create_infer_request() for _ in range(num_requests)]
        log.info('The {} model {} is loaded to {}'.format(model_type, model_path, device))

        self.input_tensor_name = self.model.inputs[0].get_any_name()

        if len(self.model.outputs) > 1:
            if output_shape is not None:
                candidates = []
                for output_tensor in self.model.outputs:
                    if len(output_tensor.partial_shape) != len(output_shape):
                        continue

                    if output_tensor.partial_shape[1] == output_shape[1]:
                        candidates.append(output_tensor.get_any_name())

                if len(candidates) != 1:
                    raise RuntimeError("One output is expected")
                self.output_tensor_name = candidates[0]
            else:
                raise RuntimeError("One output is expected")
        else:
            self.output_tensor_name = self.model.outputs[0].get_any_name()

        self.input_size = self.model.input(self.input_tensor_name).shape

    def infer(self, data):
        """Runs model on the specified input"""

        input_data = {self.input_tensor_name: data}
        self.infer_requests[0].infer(input_data)
        return self.infer_requests[0].get_tensor(self.output_tensor_name).data[:]

    def async_infer(self, data, req_id):
        """Requests model inference for the specified input"""

        input_data = {self.input_tensor_name: data}
        self.infer_requests[req_id].start_async(inputs=input_data)

    def wait_request(self, req_id):
        """Waits for the model output by the specified request ID"""

        self.infer_requests[req_id].wait()
        return self.infer_requests[req_id].get_tensor(self.output_tensor_name).data[:]
