"""
 Copyright (C) 2020 Intel Corporation

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

import logging
import os.path as osp
import threading
from collections import deque

import cv2
import ngraph as ng
import numpy as np
from openvino.inference_engine import IENetwork

from .decoder import AssociativeEmbeddingDecoder, OpenPoseDecoder


log = logging.getLogger()


class Model:
    def __init__(self, ie, xml_file_path, bin_file_path=None,
                 size_divisor=8, target_size=None,
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
        self.exec_net = ie.load_network(network=self.net, device_name=device, config=plugin_config, num_requests=max_num_requests)

        self.image_blob_name = ''
        self.size_divisor = size_divisor
        self.target_size = target_size

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
            log.info('reshape net to {}'.format(input_shapes))
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

    @staticmethod
    def _get_inputs(net):
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
    def _resize_image(frame, size):
        return cv2.resize(frame, (size, size))

    def preprocess(self, inputs):
        img = self._resize_image(inputs[self.image_blob_name], self.target_size)
        meta = {'original_shape': inputs[self.image_blob_name].shape,
                'resized_shape': img.shape}
        h, w = img.shape[:2]
        divisor = self.size_divisor
        if w % divisor != 0 or h % divisor != 0:
            img = np.pad(img, ((0, (h + divisor - 1) // divisor * divisor - h),
                               (0, (w + divisor - 1) // divisor * divisor - w),
                               (0, 0)),
                         mode='constant', constant_values=0)
        # Change data layout from HWC to CHW
        img = img.transpose((2, 0, 1))
        inputs[self.image_blob_name] = img[None]
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


class HPEOpenPose(Model):

    def __init__(self, *args, upsample_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.pooled_heatmaps_blob_name = 'pooled_heatmaps'
        self.heatmaps_blob_name = 'heatmaps'
        self.pafs_blob_name = 'pafs'

        function = ng.function_from_cnn(self.net)
        paf = function.get_output_op(0)
        paf = paf.inputs()[0].get_source_output().get_node()
        paf.set_friendly_name(self.pafs_blob_name)
        heatmap = function.get_output_op(1)
        heatmap = heatmap.inputs()[0].get_source_output().get_node()
        heatmap.set_friendly_name(self.heatmaps_blob_name)

        # Add keypoints NMS to the network.
        # Heuristic NMS kernel size adjustment depending on the feature maps upsampling ratio.
        p = int(np.round(6 / 7 * upsample_ratio))
        k = 2 * p + 1
        pooled_heatmap = ng.max_pool(heatmap, kernel_shape=(k, k), pads_begin=(p, p), pads_end=(p, p),
                                     strides=(1, 1), name=self.pooled_heatmaps_blob_name)
        f = ng.impl.Function(
            [ng.result(heatmap, name=self.heatmaps_blob_name),
             ng.result(pooled_heatmap, name=self.pooled_heatmaps_blob_name),
             ng.result(paf, name=self.pafs_blob_name)],
            function.get_parameters(), 'hpe')

        self.image_blob_name = self._get_inputs(self.net)
        self.net = IENetwork(ng.impl.Function.to_capsule(f))
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=self.max_num_requests)
        self.requests = self.exec_net.requests
        self.empty_requests = deque(self.requests)

        self.num_joints = self.net.outputs[self.heatmaps_blob_name].shape[1] - 1  # The last channel is for background.
        target_size = self.net.input_info[self.image_blob_name].input_data.shape[-2]
        self.output_scale = target_size / self.net.outputs[self.heatmaps_blob_name].shape[-2]
        if self.target_size is None:
            self.target_size = target_size

        self.decoder = OpenPoseDecoder(num_joints=self.num_joints)

    @staticmethod
    def heatmap_nms(heatmaps, pooled_heatmaps):
        return heatmaps * (heatmaps == pooled_heatmaps)

    @staticmethod
    def _resize_image(frame, size):
        h = frame.shape[0]
        scale = size / h
        resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return resized_frame

    def postprocess(self, outputs, meta):
        heatmaps = outputs[self.heatmaps_blob_name]
        pafs = outputs[self.pafs_blob_name]
        pooled_heatmaps = outputs[self.pooled_heatmaps_blob_name]
        nms_heatmaps = self.heatmap_nms(heatmaps, pooled_heatmaps)
        poses, scores = self.decoder(heatmaps, nms_heatmaps, pafs)
        # Rescale poses to the original image.
        original_image_shape = meta['original_shape']
        resized_image_shape = meta['resized_shape']
        scale_x = original_image_shape[1] / resized_image_shape[1]
        scale_y = original_image_shape[0] / resized_image_shape[0]
        poses[:, :, 0] *= scale_x * self.output_scale
        poses[:, :, 1] *= scale_y * self.output_scale
        return poses, scores


class HPEAssociativeEmbedding(Model):

    def __init__(self, *args, size_divisor=32, **kwargs):
        super().__init__(*args, size_divisor=size_divisor, **kwargs)

        self.image_blob_name = self._get_inputs(self.net)
        self.heatmaps_blob_name = find_layer_by_name('heatmaps', self.net.outputs)
        self.nms_heatmaps_blob_name = find_layer_by_name('nms_heatmaps', self.net.outputs)
        self.embeddings_blob_name = find_layer_by_name('embeddings', self.net.outputs)

        self.num_joints = self.net.outputs[self.heatmaps_blob_name].shape[1]
        h, w = self.net.input_info[self.image_blob_name].input_data.shape[-2:]
        target_size = min(h, w)
        self.output_scale = target_size / self.net.outputs[self.heatmaps_blob_name].shape[-1]
        if self.target_size is None:
            self.target_size = target_size

        self.decoder = AssociativeEmbeddingDecoder(
            num_joints=self.num_joints,
            adjust=True,
            refine=True,
            delta=0.0,
            max_num_people=30,
            detection_threshold=0.1,
            tag_threshold=1,
            use_detection_val=True,
            ignore_too_much=False)

    @staticmethod
    def _resize_image(frame, size):
        h, w = frame.shape[:2]
        scale = max(size / h, size / w)
        resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return resized_frame

    def postprocess(self, outputs, meta):
        heatmaps = outputs[self.heatmaps_blob_name]
        nms_heatmaps = outputs[self.nms_heatmaps_blob_name]
        aembds = outputs[self.embeddings_blob_name]
        poses, scores = self.decoder(heatmaps, aembds, nms_heatmaps=nms_heatmaps)
        # Rescale poses to the original image.
        original_image_shape = meta['original_shape']
        resized_image_shape = meta['resized_shape']
        scale_x = original_image_shape[1] / resized_image_shape[1]
        scale_y = original_image_shape[0] / resized_image_shape[0]
        poses[:, :, 0] *= scale_x * self.output_scale
        poses[:, :, 1] *= scale_y * self.output_scale
        return poses, scores


def find_layer_by_name(name, all_outputs):
    all_names = tuple(layer_name for layer_name in all_outputs)
    suitable_layers = [layer_name for layer_name in all_outputs if layer_name.startswith(name)]
    if not suitable_layers:
        raise ValueError('Suitable layer for "{}" output is not found in {}'.format(name, all_names))

    if len(suitable_layers) > 1:
        raise ValueError('More than 1 layer matched to "{}" output: {}'.format(name, suitable_layers))

    return suitable_layers[0]
