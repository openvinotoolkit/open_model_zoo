import logging
import os.path as osp
import threading
from collections import deque

import cv2
import numpy as np

from .decoder import AssociativeEmbeddingDecoder


log = logging.getLogger()


class Model:
    def __init__(self, ie, xml_file_path, bin_file_path=None,
                 device='CPU', plugin_config={}, max_num_requests=1,
                 results=None, caught_exceptions=None):
        self.ie = ie
        log.info('Reading network from IR...')
        if bin_file_path is None:
            bin_file_path = osp.splitext(xml_file_path)[0] + '.bin'
        self.net = ie.read_network(model=xml_file_path, weights=bin_file_path)

        log.info('Loading network to plugin...')
        self.max_num_requests = max_num_requests
        self.device = device
        self.plugin_config = plugin_config
        self.input_shapes = {}
        self.exec_net = ie.load_network(network=self.net, device_name=device, config=plugin_config, num_requests=max_num_requests)

        self.requests = self.exec_net.requests
        self.empty_requests = deque(self.requests)
        self.completed_request_results = results if results is not None else []
        self.callback_exceptions = caught_exceptions if caught_exceptions is not None else {}
        self.event = threading.Event()

    def reshape_net(self, inputs):
        input_shapes = {name: data.shape for name, data in inputs.items()}
        reshape_needed = False
        for input_name, input_shape in input_shapes.items():
            blob_shape = self.net.input_info[input_name].input_data.shape
            if not np.array_equal(input_shape, blob_shape):
                reshape_needed = True
                break
        if reshape_needed:
            self.await_all()
            self.net.reshape(input_shapes)
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=self.max_num_requests)
            self.requests = self.exec_net.requests
            self.empty_requests = deque(self.requests)

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs_dict = {next(iter(self.net.input_info)): inputs}
        else:
            inputs_dict = inputs
        return inputs_dict

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
        inputs = self.unify_inputs(inputs)
        inputs, preprocessing_meta = self.preprocess(inputs)
        self.reshape_net(inputs)
        meta.update(preprocessing_meta)
        request = self.empty_requests.popleft()
        request.set_completion_callback(py_callback=self.inference_completion_callback,
                                        py_data=(request, id, meta))
        self.event.clear()
        request.async_infer(inputs=inputs)

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        self.event.wait()


class HPE(Model):

    def __init__(self, *args, labels_map=None, keep_aspect_ratio_resize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_aspect_ratio_resize = keep_aspect_ratio_resize
        self.labels_map = labels_map

        self.image_blob_name = self._get_inputs(self.net)
        self.target_size = self.net.input_info[self.image_blob_name].input_data.shape[-1]

        self.heatmaps_blob_name = find_layer_by_name('heatmaps', self.net.outputs)
        self.nms_heatmaps_blob_name = find_layer_by_name('nms_heatmaps', self.net.outputs)
        self.embeddings_blob_name = find_layer_by_name('embeddings', self.net.outputs)

        self.decorer = AssociativeEmbeddingDecoder(
            num_joints=17,
            adjust=True,
            refine=True,
            delta=0.0,
            max_num_people=30,
            nms_kernel=5,
            tag_per_joint=True,
            detection_threshold=0.1,
            tag_threshold=1,
            use_detection_val=True,
            ignore_too_much=False)

    def _get_inputs(self, net):
        image_blob_name = None
        for blob_name, blob in net.input_info.items():
            if len(blob.input_data.shape) == 4:
                image_blob_name = blob_name
            else:
                raise RuntimeError('Unsupported {}D input layer "{}". Only 2D and 4D input layers are supported'
                                   .format(len(blob.shape), blob_name))
        if image_blob_name is None:
            raise RuntimeError('Failed to identify the input for the image.')
        return image_blob_name

    @staticmethod
    def _resize_image(frame, size, keep_aspect_ratio=False):
        if not keep_aspect_ratio:
            resized_frame = cv2.resize(frame, size)
        else:
            h, w = frame.shape[:2]
            scale = max(size[1] / h, size[0] / w)
            resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return resized_frame

    def preprocess(self, inputs):
        img = self._resize_image(inputs[self.image_blob_name], (self.target_size, self.target_size), self.keep_aspect_ratio_resize)
        meta = {'original_shape': inputs[self.image_blob_name].shape,
                'resized_shape': img.shape}
        img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        inputs[self.image_blob_name] = img[None, ...]
        return inputs, meta

    def postprocess(self, outputs, meta):
        heatmaps = outputs[self.heatmaps_blob_name]
        nms_heatmaps = outputs[self.nms_heatmaps_blob_name]
        aembds = outputs[self.embeddings_blob_name]
        poses, scores = self.decorer(heatmaps, aembds, nms_heatmaps=nms_heatmaps)
        poses = poses[0]
        scores = scores[0]
        # Rescale poses to the original image.
        original_image_shape = meta['original_shape']
        resized_image_shape = meta['resized_shape']
        scale_x = original_image_shape[1] / resized_image_shape[1]
        scale_y = original_image_shape[0] / resized_image_shape[0]
        poses[:, :, 0] *= scale_x * 2
        poses[:, :, 1] *= scale_y * 2
        return poses, scores


def find_layer_by_name(name, all_outputs):
    suitable_layers = [layer_name for layer_name in all_outputs if layer_name.startswith(name)]
    if not suitable_layers:
        raise ValueError('Suitable layer for "{}" output is not found'.format(name))

    if len(suitable_layers) > 1:
        raise ValueError('More than 1 layer matched to "{}" output: {}'.format(name, suitable_layers))

    return suitable_layers[0]
