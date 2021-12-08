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

import re
import numpy as np
import logging as log
from .model_adapter import ModelAdapter, Metadata


class OvmsAdapter(ModelAdapter):
    """
    Class that allows working with models served by the OpenVINO Model Server
    """

    tf2ov_precision = {
        "DT_INT64": "I64",
        "DT_UINT64": "U64",
        "DT_FLOAT": "FP32",
        "DT_UINT32": "U32",
        "DT_INT32": "I32",
        "DT_HALF" : "FP16",
        "DT_INT16": "I16",
        "DT_INT8" : "I8",
        "DT_UINT8": "U8",
    }

    tf2np_precision = {
        "DT_INT64": np.int64,
        "DT_UINT64": np.uint64,
        "DT_FLOAT": np.float32,
        "DT_UINT32": np.uint32,
        "DT_INT32": np.int32,
        "DT_HALF" : np.float16,
        "DT_INT16": np.int16,
        "DT_INT8" : np.int8,
        "DT_UINT8": np.uint8,
    }

    @classmethod
    def parse_model_arg(cls, target_model):
        if not isinstance(target_model, str):
            raise TypeError("--model option should be str")
        # Expecting format: <address>:<port>/models/<model_name>[:<model_version>]
        pattern = re.compile(r"(\w+\.*\-*)*\w+:\d+\/models\/\w+(\:\d+)*")
        if not pattern.fullmatch(target_model):
            raise TypeError("invalid --model option format")
        [service_url, _, model] = target_model.split("/")
        model_spec = model.split(":")
        if len(model_spec) == 1:
            # model version not specified - use latest
            return service_url, model_spec[0], 0
        elif len(model_spec) == 2:
            return service_url, model_spec[0], int(model_spec[1])
        else:
            raise TypeError("invalid --model option format")


    def _is_model_available(self):
        try:
            model_status = self.client.get_model_status(self.model_name, self.model_version)
        except ovmsclient.ModelNotFoundError:
            return False
        target_version = max(model_status.keys())
        version_status = model_status[target_version]
        if version_status["state"] == "AVAILABLE" and version_status["error_code"] == 0:
            return True
        return False

    def _prepare_inputs(self, dict_data):
        inputs = {}
        for input_name, input_data in dict_data.items():
            if input_name not in self.metadata["inputs"].keys():
                raise ValueError("Input data does not match model inputs")
            input_info = self.metadata["inputs"][input_name]
            model_precision = self.tf2np_precision[input_info["dtype"]]
            if isinstance(input_data, np.ndarray) and input_data.dtype != model_precision:
                input_data = input_data.astype(model_precision)
            elif isinstance(input_data, list):
                input_data = np.array(input_data, dtype=model_precision)
            inputs[input_name] = input_data
        return inputs

    def __init__(self, target_model):
        if ovmsclient_absent:
            raise ImportError("The ovmsclient package is not installed")

        log.info('Connecting to remote model: {}'.format(target_model))
        service_url, model_name, model_version = OvmsAdapter.parse_model_arg(target_model)
        self.model_name = model_name
        self.model_version = model_version
        self.client = ovmsclient.make_grpc_client(url=service_url)
        # Ensure the model is available
        if not self._is_model_available():
            model_version_str = "latest" if self.model_version == 0 else str(self.model_version)
            raise RuntimeError("Requested model: {}, version: {}, has not been found or is not "
                "in available state".format(self.model_name, model_version_str))

        self.metadata = self.client.get_model_metadata(model_name=self.model_name,
                                                       model_version=self.model_version)

    def load_model(self):
        pass

    def get_input_layers(self):
        inputs = {}
        for name, meta in self.metadata["inputs"].items():
            inputs[name] = Metadata(meta["shape"], self.tf2ov_precision.get(meta["dtype"], meta["dtype"]))
        return inputs

    def get_output_layers(self):
        outputs = {}
        for name, meta in self.metadata["outputs"].items():
            outputs[name] = Metadata(meta["shape"], self.tf2ov_precision.get(meta["dtype"], meta["dtype"]))
        return outputs

    def reshape_model(self, new_shape):
        pass

    def infer_sync(self, dict_data):
        inputs = self._prepare_inputs(dict_data)
        raw_result = self.client.predict(inputs, model_name=self.model_name, model_version=self.model_version)
        # For models with single output ovmsclient returns ndarray with results,
        # so the dict must be created to correctly implement interface.
        if isinstance(raw_result, np.ndarray):
            output_name = list(self.metadata["outputs"].keys())[0]
            return {output_name: raw_result}
        return raw_result

    def infer_async(self, dict_data, callback_fn, callback_data):
        inputs = self._prepare_inputs(dict_data)
        raw_result = self.client.predict(inputs, model_name=self.model_name, model_version=self.model_version)
        # For models with single output ovmsclient returns ndarray with results,
        # so the dict must be created to correctly implement interface.
        if isinstance(raw_result, np.ndarray):
            output_name = list(self.metadata["outputs"].keys())[0]
            raw_result = {output_name: raw_result}
        callback_fn(0, (lambda x: x, raw_result, callback_data))

    def is_ready(self):
        return True

    def await_all(self):
        pass

    def await_any(self):
        pass
