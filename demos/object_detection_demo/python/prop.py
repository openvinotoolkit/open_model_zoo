import collections
import logging as log
from pathlib import Path

try:
    from openvino.runtime import AsyncInferQueue, Core, PartialShape, layout_helpers, get_version
    openvino_absent = False
except ImportError:
    openvino_absent = True

from openvino.model_zoo.model_api.adapters.model_adapter import Metadata
from openvino.model_zoo.model_api.adapters.utils import Layout
from openvino.model_zoo.model_api.pipelines import parse_devices

class OvInferrer:
    def __init__(self, core, model_path, weights_path=None, model_parameters = {}, device='CPU', plugin_config=None, nireq=0):
        self.core = core
        self.model_path = model_path
        self.device = device
        self.plugin_config = plugin_config
        self.nireq = nireq
        self.model_parameters = model_parameters

        self.empty_ireqs = []
        self.busy_ireqs = collections.deque()
        self.stop_submit = False

        if isinstance(model_path, (str, Path)):
            if Path(model_path).suffix == ".onnx" and weights_path:
                log.warning('For model in ONNX format should set only "model_path" parameter.'
                            'The "weights_path" will be omitted')

        self.model_from_buffer = isinstance(model_path, bytes) and isinstance(weights_path, bytes)
        log.info('Reading model {}'.format('from buffer' if self.model_from_buffer else model_path))
        weights = weights_path if self.model_from_buffer else ''
        self.model = core.read_model(model_path, weights)

    def __del__(self):
        for ireq in self.busy_ireqs:
            ireq[0].cancel()

    def load_model(self):
        self.compiled_model = self.core.compile_model(self.model, self.device, self.plugin_config)
        async_queue = AsyncInferQueue(self.compiled_model, self.nireq)
        if self.nireq == 0:
            # +1 to use it as a buffer of the pipeline
            async_queue = AsyncInferQueue(self.compiled_model, len(async_queue) + 1)

        self.empty_ireqs = [self.compiled_model.create_infer_request() for _ in range(len(async_queue))]  # TODO: ask nireq normally
        self.busy_ireqs = collections.deque(maxlen=len(self.empty_ireqs))
        log.info('The model {} is loaded to {}'.format("from buffer" if self.model_from_buffer else self.model_path, self.device))
        self.log_runtime_settings()

    def log_runtime_settings(self):
        devices = set(parse_devices(self.device))
        if 'AUTO' not in devices:
            for device in devices:
                try:
                    nstreams = self.compiled_model.get_property(device + '_THROUGHPUT_STREAMS')
                    log.info('\tDevice: {}'.format(device))
                    log.info('\t\tNumber of streams: {}'.format(nstreams))
                    if device == 'CPU':
                        nthreads = self.compiled_model.get_property('CPU_THREADS_NUM')
                        log.info('\t\tNumber of threads: {}'.format(nthreads if int(nthreads) else 'AUTO'))
                except RuntimeError:
                    pass
        log.info('\tNumber of model infer requests: {}'.format(len(self.empty_ireqs)))

    def get_input_layers(self):
        inputs = {}
        self.model_parameters['input_layouts'] = Layout.parse_layouts(self.model_parameters.get('input_layouts', None))
        for input in self.model.inputs:
            input_layout = self.get_layout_for_input(input)
            inputs[input.get_any_name()] = Metadata(input.get_names(), list(input.shape), input_layout, input.get_element_type().get_type_name())
        inputs = self._get_meta_from_ngraph(inputs)
        return inputs

    def get_layout_for_input(self, input) -> str:
        input_layout = ''
        if self.model_parameters['input_layouts']:
            input_layout = Layout.from_user_layouts(input.get_names(), self.model_parameters['input_layouts'])
        if not input_layout:
            if not layout_helpers.get_layout(input).empty:
                input_layout = Layout.from_openvino(input)
            elif len(input.shape) == 4:
                input_layout = Layout.from_shape(input.shape)
        return input_layout

    def get_output_layers(self):
        outputs = {}
        for output in self.model.outputs:
            output_shape = output.partial_shape.get_min_shape() if self.model.is_dynamic() else output.shape
            outputs[output.get_any_name()] = Metadata(output.get_names(), list(output_shape), precision=output.get_element_type().get_type_name())
        outputs = self._get_meta_from_ngraph(outputs)
        return outputs

    def reshape_model(self, new_shape):
        new_shape = {k: PartialShape(v) for k, v in new_shape.items()}
        self.model.reshape(new_shape)

    def get_raw_result(self, request):
        return {key: request.get_tensor(key).data[:] for key in self.get_output_layers().keys()}

    def _get_meta_from_ngraph(self, layers_info):
        for node in self.model.get_ordered_ops():
            layer_name = node.get_friendly_name()
            if layer_name not in layers_info.keys():
                continue
            layers_info[layer_name].meta = node.get_attributes()
            layers_info[layer_name].type = node.get_type_name()
        return layers_info

    def operations_by_type(self, operation_type):
        layers_info = {}
        for node in self.model.get_ordered_ops():
            if node.get_type_name() == operation_type:
                layer_name = node.get_friendly_name()
                layers_info[layer_name] = Metadata(type=node.get_type_name(), meta=node.get_attributes())
        return layers_info

    class Iterate:
        def __init__(self, inferrer):
            self.inferrer = inferrer
        def __iter__(self):
            return self
        def __next__(self):
            if not self.inferrer.stop_submit or self.inferrer.busy_ireqs:
                return self.inferrer.state()
            raise StopIteration

    def end(self):
        if self.stop_submit:
            raise RuntimeError("Input was over. Unexpected end")
        self.stop_submit = True

    def submit(self, input, meta):
        if self.stop_submit:
            raise RuntimeError("Input was over. Unexpected submit")
        ireq = self.empty_ireqs.pop()
        ireq.start_async(input)
        self.busy_ireqs.append((ireq, meta))

    def state(self):
        if self.busy_ireqs and self.busy_ireqs[0][0].wait_for(0):
            ireq, meta = self.busy_ireqs.popleft()
            self.empty_ireqs.append(ireq)
            return self.get_raw_result(ireq), meta
        if not self.stop_submit and self.empty_ireqs:
            return None
        self.busy_ireqs[0][0].wait()
        ireq, meta = self.busy_ireqs.popleft()
        self.empty_ireqs.append(ireq)
        return self.get_raw_result(ireq), meta
