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

import ovmsclient

from .model_adapter import ModelAdapter


class RemoteAdapter(ModelAdapter):
    """
    Class that allows working with Remote OpenVino Model Server model
    """

    def __init__(self, model_name, config):
        self.model_name = str(model_name)
        self.load_model(config)

    def load_model(self, config):
        self.client = ovmsclient.make_grpc_client(config=config)
        # ensure the model can be loaded
        ovmsclient.make_grpc_status_request(model_name=self.model_name)

        metadata_request = ovmsclient.make_grpc_metadata_request(model_name=self.model_name)
        metadata_response = self.client.get_model_metadata(metadata_request)
        self.metadata = metadata_response.to_dict()[1]

    def get_input_layers(self):
        return list(self.metadata["inputs"].keys())

    def get_output_layers(self):
        return list(self.metadata["outputs"].keys())

    def get_input_layer_shape(self, input_layer_name):
        return self.metadata["inputs"][input_layer_name]['shape']

    def get_output_layer_shape(self, output_layer_name):
        return self.metadata["outputs"][output_layer_name]['shape']

    def get_input_layer_precision(self, input_layer_name):
        return self.metadata["inputs"][input_layer_name]['dtype']

    def get_output_layer_precision(self, output_layer_name):
        return self.metadata["outputs"][output_layer_name]['dtype']

    def create_infer_request(self, input_layer_name, data):
        return ovmsclient.make_grpc_predict_request(
            {input_layer_name: data}, model_name=self.model_name
        )

    def infer(self, infer_request):
        return self.client.predict(infer_request).to_dict()

    def reshape_model(self, new_shape):
        pass
