"""
Copyright (c) 2018-2021 Intel Corporation

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
        if completion_callback:
            self.request.set_completion_callback(completion_callback, self.request_id)
        self.context = None
        self._contains_blob = hasattr(self.request, 'output_blobs')

    def infer(self, inputs, meta, context=None):
        if context:
            self.context = context
        self.meta = meta
        self.request.async_infer(inputs=inputs)

    def get_result(self):
        if not self._contains_blob:
            return self.context, self.meta, self.request.outputs
        outputs = OrderedDict()
        for output_name, output_blob in self.request.output_blobs.items():
            outputs[output_name] = output_blob.buffer
        return self.context, self.meta, outputs

    def set_completion_callback(self, callback):
        self.request.set_completion_callback(callback, self.request_id)
