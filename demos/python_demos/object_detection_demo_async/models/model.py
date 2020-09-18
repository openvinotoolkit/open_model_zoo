from collections import deque
import threading
from time import perf_counter


class Model:
    def __init__(self, ie, model_path, logger=None, device='CPU', plugin_config={}, max_num_requests=1,
                 results=None, caught_exceptions=None):
        self.logger = logger
        if logger:
            logger.info('Reading network from IR...')
        loading_time = perf_counter()
        self.net = ie.read_network(model_path, model_path[:-4] + '.bin')
        loading_time = (perf_counter() - loading_time)
        if logger:
            logger.info('Read in {:.3f} seconds'.format(loading_time))

        if logger:
            logger.info('Loading network to {} plugin...'.format(device))
        loading_time = perf_counter()
        self.max_num_requests = max_num_requests
        self.exec_net = ie.load_network(network=self.net, device_name=device,
                                        config=plugin_config, num_requests=max_num_requests)
        loading_time = (perf_counter() - loading_time)
        if logger:
            logger.info('Loaded in {:.3f} seconds'.format(loading_time))

        self.requests = self.exec_net.requests
        self.empty_requests = deque(self.requests)
        self.completed_request_results = results if results is not None else []
        self.callback_exceptions = caught_exceptions if caught_exceptions is not None else {}
        self.event = threading.Event()

    def unify_inputs(self, inputs) -> dict:
        raise NotImplementedError

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

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
        inputs = self.unify_inputs(inputs)
        inputs, preprocessing_meta = self.preprocess(inputs)
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
