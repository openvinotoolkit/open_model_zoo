import threading
from collections import deque
from time import perf_counter

from .model import Model


class SyncModelRunner:
    def __init__(self, ie, model, device='CPU'):
        self.device = device
        self.model = model

        self.exec_net = ie.load_network(network=model, device_name=self.device)

    def __call__(self, inputs):
        inputs, meta = self.model.preprocess(inputs)
        outputs = self.exec_net.infer(inputs=inputs)
        outputs = self.model.postprocess(outputs, meta)
        return outputs


class AsyncModelRunner:
    def __init__(self, ie, model, device='CPU', plugin_config={}, max_num_requests=1,
                 results=None, caught_exceptions=None):
        self.model = model
        self.device = device
        verbose = True
        if verbose:
            print('Loading network to {} plugin...'.format(device))
        loading_time = perf_counter()
        self.max_num_requests = max_num_requests
        self.exec_net = ie.load_network(network=self.net, device_name=device,
                                        config=plugin_config, num_requests=max_num_requests)
        loading_time = (perf_counter() - loading_time) / cv2.getTickFrequency()
        if verbose:
            print('Loaded in {:.3f} seconds'.format(loading_time))

        self.requests = self.exec_net.requests
        self.empty_requests = deque(self.requests)
        self.completed_request_results = results if results is not None else []
        self.callback_exceptions = caught_exceptions if caught_exceptions is not None else {}
        self.event = threading.Event()

    def inference_completion_callback(self, status, callback_args):
        request, frame_id, frame_meta = callback_args
        try:
            if status != 0:
                raise RuntimeError('Infer Request has returned status code {}'.format(status))
            raw_outputs = {key: blob.buffer for key, blob in request.output_blobs.items()}
            self.completed_request_results[frame_id] = (frame_meta, raw_outputs)
            self.empty_requests.append(request)
        except Exception as e:
            self.callback_exceptions.append(e)
        self.event.set()

    def __call__(self, inputs, id, meta):
        request = self.empty_requests.popleft()
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        meta.update(preprocessing_meta)
        request.set_completion_callback(py_callback=self.inference_completion_callback,
                                        py_data=(request, id, meta))
        self.event.clear()
        request.async_infer(inputs=inputs)

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        self.event.wait()
