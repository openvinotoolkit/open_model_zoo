#!/usr/bin/env python3
"""
 Copyright (C) 2018-2020 Intel Corporation

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

import colorsys
import logging as log
import os
import os.path as osp
import random
import sys
import time

import cv2
import numpy as np
from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IECore

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))
import monitors


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', help='Show this help message and exit.', action='help', default=SUPPRESS)
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=str)
    args.add_argument('-i', '--input', help='Required. Path to image or video file or index of camera.',
                      required=True, type=str)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU',
                      type=str, default='CPU')
    args.add_argument('--labels', help='Optional. Path to labels mapping file', type=str, default=None)
    args.add_argument('-pt', '--prob_threshold', help='Optional. Probability threshold for detections filtering',
                      type=float, default=0.5)
    args.add_argument('--no_show', help="Optional. Don't show output", action='store_true')
    args.add_argument('-u', '--utilization_monitors', type=str, default='',
                      help='Optional. List of monitors to show initially.')
    args.add_argument('--keep_aspect_ratio', help='Optional. Force image resize to keep aspect ratio.',
                      action='store_true')
    args.add_argument('--delay',
                      help='Optional. Interval in milliseconds of waiting for a key to be pressed.',
                      type=int, default=1)

    return parser


class ColorPalette:
    def __init__(self, n, seed=0xABC):
        n = int(n)
        assert n > 0

        if seed is not None:
            random.seed(seed)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for i in range(1, n):
            colors_candidates = [(random.random(), random.uniform(0.8, 1.0), random.uniform(0.5, 1.0)) 
                                for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @staticmethod
    def min_distance(colors_set, color_candidate):
        distances = [__class__.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return tuple(self.palette[n % len(self.palette)])


class Model:
    def __init__(self, ie, xml_file_path, bin_file_path=None,
                 device='CPU', max_num_requests=1):
        self.ie = ie
        log.info('Reading network from IR...')
        if bin_file_path is None:
            bin_file_path = osp.splitext(xml_file_path)[0] + '.bin'
        self.net = ie.read_network(model=xml_file_path, weights=bin_file_path)

        log.info('Loading network to plugin...')
        if 'CPU' in device:
            self.check_cpu_support(ie, self.net)
        self.max_num_requests = max_num_requests
        self.exec_net = ie.load_network(network=self.net, device_name=device, num_requests=max_num_requests)

        self.is_async = False
        self.meta = {}

    @staticmethod
    def check_cpu_support(ie, net):
        log.info('Check that all layers are supported...')
        supported_layers = ie.query_network(net, 'CPU')
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            unsupported_info = '\n\t'.join('{} ({} with params {})'.format(layer_id,
                                                                           net.layers[layer_id].type,
                                                                           str(net.layers[layer_id].params))
                                           for layer_id in not_supported_layers)
            log.warning('Following layers are not supported '
                        'by the CPU plugin:\n\t{}'.format(unsupported_info))
            log.warning('Please try to specify cpu extensions library path.')
            raise ValueError('Some of the layers are not supported.')

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            if len(inputs) == 1:
                inputs_dict = {next(iter(self.net.inputs.keys())): inputs[0]}
            else:
                raise ValueError
        else:
            inputs_dict = inputs
        return inputs_dict

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

    def __call__(self, inputs, request_id=0):
        inputs = self.unify_inputs(inputs)
        inputs, meta = self.preprocess(inputs)
        self.meta[request_id] = meta
        self.exec_net.start_async(request_id=request_id, inputs=inputs)

    def get_output(self, request_id=0):
        if self.exec_net.requests[request_id].wait(-1) == 0:
            outputs = self.exec_net.requests[request_id].outputs
            outputs = self.postprocess(outputs, self.meta[request_id])
            return outputs
        raise RuntimeError


class Detector(Model):
    def __init__(self, *args, labels_map=None, keep_aspect_ratio_resize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_aspect_ratio_resize = keep_aspect_ratio_resize
        self.labels_map = labels_map

        self.image_blob_name, self.image_info_blob_name = self._get_inputs(self.net)
        self.n, self.c, self.h, self.w = self.net.inputs[self.image_blob_name].shape
        assert self.n == 1, 'Only batch size == 1 is supported.'

        self.output_parser = self._get_output_parser(self.net, self.image_blob_name)

    def _get_inputs(self, net):
        image_blob_name = None
        image_info_blob_name = None
        for blob_name, blob in net.inputs.items():
            if len(blob.shape) == 4:
                image_blob_name = blob_name
            elif len(blob.shape) == 2:
                image_info_blob_name = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                   .format(len(blob.shape), blob_name))
        assert image_blob_name is not None
        return image_blob_name, image_info_blob_name

    def _get_output_parser(self, net, image_blob_name, bboxes='bboxes', labels='labels', scores='scores'):
        if len(net.outputs) == 1:
            output_blob = next(iter(net.outputs))
            return SingleOutputParser(output_blob)
        else:
            try:
                parser = MultipleOutputParser(net.outputs, bboxes, scores, labels)
            except ValueError:
                h, w = net.inputs[image_blob_name].shape[2:]
                parser = OTEParser([w, h], net.outputs)
            return parser
        raise RuntimeError('Unsupported model outputs')

    @staticmethod
    def _resize_image(frame, size, keep_aspect_ratio=False):
        if not keep_aspect_ratio:
            resized_frame = cv2.resize(frame, size)
        else:
            h, w = frame.shape[:2]
            scale = min(size[1] / h, size[0] / w)
            resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return resized_frame

    def preprocess(self, inputs):
        img = self._resize_image(inputs[self.image_blob_name], (self.w, self.h), self.keep_aspect_ratio_resize)
        h, w = img.shape[:2]
        if self.image_info_blob_name is not None:
            inputs[self.image_info_blob_name] = [h, w, 1]
        meta = {'original_shape': inputs[self.image_blob_name].shape,
                'resized_shape': img.shape}
        if h != self.h or w != self.w:
            img = np.pad(img, ((0, self.h - h), (0, self.w - w), (0, 0)),
                         mode='constant', constant_values=0)
        img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        img = img.reshape((self.n, self.c, self.h, self.w))
        inputs[self.image_blob_name] = img
        return inputs, meta

    def postprocess(self, outputs, meta):
        detections = self.output_parser(outputs)
        orginal_image_shape = meta['original_shape']
        resized_image_shape = meta['resized_shape']
        scale_x = self.w / resized_image_shape[1] * orginal_image_shape[1]
        scale_y = self.h / resized_image_shape[0] * orginal_image_shape[0]
        detections[:, 3:7:2] *= scale_x
        detections[:, 4:7:2] *= scale_y
        return detections


class SingleOutputParser:
    def __init__(self, output_layer):
        self.output_layer = output_layer

    def __call__(self, outputs):
        return np.asarray(outputs[self.output_layer][0][0])


class MultipleOutputParser:
    def __init__(self, all_outputs, bboxes_layer='bboxes', scores_layer='scores', labels_layer='labels'):
        self.labels_layer = self.find_layer(labels_layer, all_outputs)
        self.scores_layer = self.find_layer(scores_layer, all_outputs)
        self.bboxes_layer = self.find_layer(bboxes_layer, all_outputs)

    @staticmethod
    def find_layer(name, all_outputs):
        suitable_layers = [layer_name for layer_name in all_outputs if name in layer_name]
        if not suitable_layers:
            raise ValueError('Suitable layer for "{}" output is not found'.format(name))

        if len(suitable_layers) > 1:
            raise ValueError('More than 1 layer matched to "{}" output'.format(name))

        return suitable_layers[0]

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer][0]
        scores = outputs[self.scores_layer][0]
        labels = outputs[self.labels_layer][0]
        return np.asarray([[0, label, score, *bbox] for label, score, bbox in zip(labels, scores, bboxes)])


class OTEParser:
    def __init__(self, input_size, all_outputs, labels_layer='labels', default_label=1):
        try:
            self.labels_layer = self.find_layer_by_name(labels_layer, all_outputs)
            log.info('Use output "{}" as the one containing labels of detected objects.'
                     .format(self.labels_layer))
        except ValueError:
            log.warning('Suitable layer for "{}" output is not found. '
                        'Treating detector as class agnostic with output label {}'
                        .format(labels_layer, default_label))
            self.labels_layer = None
            self.default_label = default_label
            log.info('Treating detector as a class-agnostic one with output label {}.'.format(self.default_label))

        self.input_size = input_size
        self.bboxes_layer = self.find_layer_bboxes_output(all_outputs)
        log.info('Use auto-detected output "{}" as the one containing detected bounding boxes.'
                 .format(self.bboxes_layer))

    @staticmethod
    def find_layer_bboxes_output(all_outputs):
        filter_outputs = [
            output_name for output_name, out_data in all_outputs.items()
            if len(np.shape(out_data)) == 2 and np.shape(out_data)[-1] == 5
        ]
        if not filter_outputs:
            raise ValueError('Suitable output with bounding boxes is not found')
        if len(filter_outputs) > 1:
            raise ValueError('More than 1 candidate for output with bounding boxes.')
        return filter_outputs[0]

    @staticmethod
    def find_layer_by_name(name, all_outputs):
        return MultipleOutputParser.find_layer(name, all_outputs)

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer]
        scores = bboxes[:, 4]
        bboxes = bboxes[:, :4]
        bboxes[:, 0::2] /= self.input_size[0]
        bboxes[:, 1::2] /= self.input_size[1]
        if self.labels_layer:
            labels = outputs[self.labels_layer] + 1
        else:
            labels = np.full(len(bboxes), self.default_label, dtype=bboxes.dtype)
        return np.asarray([[0, label, score, *bbox] for label, score, bbox in zip(labels, scores, bboxes)])


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    labels_map = None
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]

    log.info('Loading network...')
    detector = Detector(ie, args.model, device=args.device, max_num_requests=2,
                        labels_map=labels_map, keep_aspect_ratio_resize=args.keep_aspect_ratio)
    detector.is_async = False

    try:
        input_stream = int(args.input)
    except ValueError:
        input_stream = args.input
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + str(input_stream)

    palette = ColorPalette(len(labels_map) if labels_map is not None else 100)
    presenter = monitors.Presenter(args.utilization_monitors, 45,
        (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))

    cur_request_id = 0
    next_request_id = 1

    if detector.is_async:
        ret, frame = cap.read()
        frame_h, frame_w = frame.shape[:2]

    log.info('Starting inference in sync mode...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")

    render_time = 0
    while cap.isOpened():
        if detector.is_async:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
            if ret:
                frame_h, frame_w = frame.shape[:2]
        if not ret:
            break  # abandons the last frame in case of async_mode
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        if detector.is_async:
            detector(next_frame, request_id=next_request_id)
        else:
            detector(frame, request_id=cur_request_id)

        outputs = detector.get_output(cur_request_id)
        inf_end = time.time()
        det_time = inf_end - inf_start

        # Parse detection results of the current request
        for obj in outputs:
            # Draw only objects when probability more than specified threshold
            if obj[2] > args.prob_threshold:
                xmin = int(obj[3])
                ymin = int(obj[4])
                xmax = int(obj[5])
                ymax = int(obj[6])
                class_id = int(obj[1])
                # Draw box and label\class_id
                color = palette[class_id]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                det_label = labels_map[class_id] if labels_map else str(class_id)
                cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        # Draw performance stats
        inf_time_message = r'Inference time: N\A for async mode' if detector.is_async else \
            'Inference time: {:.3f} ms'.format(det_time * 1000)
        render_time_message = 'OpenCV rendering time: {:.3f} ms'.format(render_time * 1000)
        async_mode_message = 'Async mode is on. Processing request {}'.format(cur_request_id) if detector.is_async else \
            'Async mode is off. Processing request {}'.format(cur_request_id)

        cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        cv2.putText(frame, async_mode_message, (10, int(frame_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (10, 10, 200), 1)

        presenter.drawGraphs(frame)
        render_start = time.time()
        if not args.no_show:
            cv2.imshow('Detection Results', frame)
        render_end = time.time()
        render_time = render_end - render_start

        if detector.is_async:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
            frame_h, frame_w = frame.shape[:2]

        ESC_KEY = 27
        TAB_KEY = 9
        if not args.no_show:
            key = cv2.waitKey(args.delay)
            if key == ESC_KEY:
                break
            if key == TAB_KEY:
                detector.is_async = not detector.is_async
                log.info('Switched to {} mode'.format('async' if detector.is_async else 'sync'))
            else:
                presenter.handleKey(key)
    print(presenter.reportMeans())


if __name__ == '__main__':
    sys.exit(main() or 0)
