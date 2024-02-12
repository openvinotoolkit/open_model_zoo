"""
Copyright (c) 2018-2024 Intel Corporation

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

from collections import OrderedDict


class AsyncInferRequestWrapper:
    def __init__(self, request_id, request, completion_callback=None):
        self.request_id = request_id
        self.request = request
        self._contains_tensors = hasattr(self.request, 'output_tensors')
        if completion_callback:
            if not self._contains_tensors:
                self.request.set_completion_callback(completion_callback, self.request_id)
            else:
                self.request.set_callback(completion_callback, self.request_id)
        self.context = None
        self._contains_blob = hasattr(self.request, 'output_blobs')

    def infer(self, inputs, meta, context=None):
        if context:
            self.context = context
        self.meta = meta
        if hasattr(self.request, 'async_infer'):
            self.request.async_infer(inputs=inputs)
        else:
            self.request.start_async(inputs=inputs)
            self.request.wait()

    def get_result(self):
        if not self._contains_blob and not self._contains_tensors:
            return self.context, self.meta, self.request.outputs
        if self._contains_tensors:
            return self.context, self.meta, {
                out.get_node().friendly_name: res.data
                for out, res in zip(self.request.outputs, self.request.output_tensors)
            }
        outputs = OrderedDict()
        for output_name, output_blob in self.request.output_blobs.items():
            outputs[output_name] = output_blob.buffer
        return self.context, self.meta, outputs

    def set_completion_callback(self, callback):
        if not self._contains_tensors:
            self.request.set_completion_callback(callback, self.request_id)
        else:
            self.request.set_callback(callback, self.request_id)
