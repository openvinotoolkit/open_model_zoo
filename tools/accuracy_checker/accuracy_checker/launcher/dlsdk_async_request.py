class AsyncInferRequestWrapper:
    def __init__(self, request_id, request, completion_callback=None):
        self.request_id = request_id
        self.request = request
        if completion_callback:
            self.request.set_completion_callback(completion_callback, self.request_id)
        self.context = None

    def infer(self, inputs, meta, context=None):
        if context:
            self.context = context
        self.meta = meta
        self.request.async_infer(inputs=inputs)

    def get_result(self):
        return self.context, self.meta, self.request.outputs

    def set_completion_callback(self, callback):
        self.request.set_completion_callback(callback, self.request_id)
