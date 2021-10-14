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


class ModelExecutor:
    def __init__(self):
        pass

    def get_input_layers(self):
        raise NotImplementedError

    def get_output_layers(self):
        raise NotImplementedError

    def get_input_layer_shape(self, layer_name):
        raise NotImplementedError

    def get_output_layer_shape(self, layer_name):
        raise NotImplementedError

    def get_input_layer_precision(self, layer_name):
        raise NotImplementedError

    def get_output_layer_precision(self, layer_name):
        raise NotImplementedError

    def create_infer_request(self, layer_name, data):
        raise NotImplementedError


class InferenceEngineExecutor(ModelExecutor):
    """
    Class that allows working with Inference Engine model, its input and output blobs
    """

    def __init__(self, ie, model_path, plugin_config, device, max_num_requests=1):
        self.net = ie.read_network(model_path)
        self.inputs = self.net.input_info
        self.outputs = self.net.outputs

        self.exec_net = ie.load_network(network=self.net, device_name=device,
            config=plugin_config, num_requests=max_num_requests)
        if max_num_requests == 0:
            # ExecutableNetwork doesn't allow creation of additional InferRequests. Reload ExecutableNetwork
            # +1 to use it as a buffer of the pipeline
            self.exec_net = ie.load_network(network=self.net, device_name=device,
                config=plugin_config, num_requests=len(self.exec_net.requests) + 1)

    def get_input_layers(self):
        return list(self.inputs.keys())

    def get_output_layers(self):
        return list(self.outputs.keys())

    def get_input_layer_shape(self, layer_name):
        return self.inputs[layer_name].input_data.shape

    def get_output_layer_shape(self, layer_name):
        return self.outputs[layer_name].shape

    def get_input_layer_precision(self, layer_name):
        return self.inputs[layer_name].precision

    def get_output_layer_precision(self, layer_name):
        return self.outputs[layer_name].precision

    def create_infer_request(self, layer_name, data):
        return {layer_name: data}


class RemoteExecutor(ModelExecutor):
    """
    Class that allows working with Remote OpenVino Model Server model
    """
    def __init__(self, model_name, serving_config):
        pass
