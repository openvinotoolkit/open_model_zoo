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

try:
    import ovmsclient
    ovmsclient_absent = False
except ImportError:
    ovmsclient_absent = True

from .model_adapter import ModelAdapter, Metadata


class RemoteAdapter(ModelAdapter):
    """
    Class that allows working with Remote OpenVino Model Server model
    """

    precisions = {
        'DT_FLOAT': 'FP32',
        'DT_INT32': 'I32',
        'DT_HALF' : 'FP16',
        'DT_INT16': 'I16',
        'DT_INT8' : 'I8',
        'DT_UINT8': 'U8',
    }

    def __init__(self, model_name, config):
        if ovmsclient_absent:
            raise ImportError('The OVMSclient package is not installed')

        self.model_name = model_name
        self.client = ovmsclient.make_grpc_client(config=config)
        # ensure the model can be loaded
        ovmsclient.make_grpc_status_request(model_name=self.model_name)

        metadata_request = ovmsclient.make_grpc_metadata_request(model_name=self.model_name)
        self.metadata = self.client.get_model_metadata(metadata_request).to_dict()[1]

    def load_model(self):
        pass

    def get_input_layers(self):
        inputs = {}
        for name, meta in self.metadata['inputs'].items():
            inputs[name] = Metadata(meta['shape'], self.precisions.get(meta['dtype'], meta['dtype']))
        return inputs

    def get_output_layers(self):
        outputs = {}
        for name, meta in self.metadata['outputs'].items():
            outputs[name] = Metadata(meta['shape'], self.precisions.get(meta['dtype'], meta['dtype']))
        return outputs

    def reshape_model(self, new_shape):
        pass

    def infer_sync(self, dict_data):
        predict_request = ovmsclient.make_grpc_predict_request(
            dict_data, model_name=self.model_name)
        return self.client.predict(predict_request).to_dict()

    def infer_async(self, dict_data, callback_fn, callback_data):
        raw_result = self.infer_sync(dict_data)
        callback_fn(0, (lambda x: x, raw_result, callback_data))

    def is_ready(self):
        return True

    def await_all(self):
        pass

    def await_any(self):
        pass
