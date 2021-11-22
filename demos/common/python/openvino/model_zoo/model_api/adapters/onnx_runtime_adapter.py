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
    import onnxruntime as ort
    onnxruntime_absent = False
except ImportError:
    onnxruntime_absent = True

from .model_adapter import ModelAdapter, Metadata


class ONNXRuntimeAdapter(ModelAdapter):
    precisions = {
        'float': 'FP32',
        'int64': 'I32'
    }

    def __init__(self, model):
        if onnxruntime_absent:
            raise ImportError('Can not find the ONNX Runtime package. Please install')
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.inference_session = ort.InferenceSession(model, sess_options=session_options)
        self.inputs = list(self.get_input_layers().keys())
        self.outputs = list(self.get_output_layers().keys())

    def load_model(self):
        pass

    def get_input_layers(self):
        nodes = self.inference_session.get_inputs()
        inputs = {}
        for node in nodes:
            precision = self.precisions[node.type.replace('tensor(', '').replace(')', '')]
            inputs[node.name] = Metadata(shape=node.shape, precision=precision)
        return inputs

    def get_output_layers(self):
        nodes = self.inference_session.get_outputs()
        outputs = {}
        for node in nodes:
            precision = self.precisions[node.type.replace('tensor(', '').replace(')', '')]
            outputs[node.name] = Metadata(shape=node.shape, precision=precision)
        return outputs

    def infer_sync(self, dict_data):
        raw_results = self.inference_session.run(self.outputs, dict_data)
        return dict(zip(self.outputs, raw_results))

    def infer_async(self, dict_data, callback_fn, callback_data):
        raw_results = self.inference_session.run(self.outputs, dict_data)
        results = dict(zip(self.outputs, raw_results))
        callback_fn(0, (lambda x: x, results, callback_data))

    def reshape_model(self, new_shape):
        pass

    def await_any(self):
        pass

    def await_all(self):
        pass

    def is_ready(self):
        return True
