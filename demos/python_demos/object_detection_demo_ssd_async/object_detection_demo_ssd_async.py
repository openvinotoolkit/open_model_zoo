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
import logging
import threading
import os.path as osp
import random
import sys
from argparse import ArgumentParser, SUPPRESS
from collections import deque, namedtuple
from itertools import cycle
from enum import Enum
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'common'))
import monitors


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=str)
    args.add_argument('-i', '--input', help='Required. Path to an image, video file or a numeric camera ID.',
                      required=True, type=str)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.', default='CPU', type=str)
    args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    args.add_argument('-t', '--prob_threshold', help='Optional. Probability threshold for detections filtering.',
                      default=0.5, type=float)
    args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                      default=False, action='store_true')
    args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                      default=1, type=int)
    args.add_argument('-nstreams', '--num_streams',
                      help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode '
                           '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                           'or just <nstreams>)',
                      default='', type=str)
    args.add_argument('-nthreads', '--num_threads',
                      help='Optional. Number of threads to use for inference on CPU (including HETERO cases)',
                      default=None, type=int)
    args.add_argument('-loop_input', '--loop_input', help='Optional. Number of times to repeat the input.',
                      type=int, default=0)
    args.add_argument('-no_show', '--no_show', help="Optional. Don't show output", action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    args.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                      help='Optional. Keeps aspect ratio on resize.')
    return parser


class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0)) 
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
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


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
        if 'CPU' in device:
            self.check_cpu_support(ie, self.net)
        self.max_num_requests = max_num_requests
        self.exec_net = ie.load_network(network=self.net, device_name=device, config=plugin_config, num_requests=max_num_requests)

        self.requests = self.exec_net.requests
        self.empty_requests = deque(self.requests)
        self.completed_request_results = results if results is not None else []
        self.callback_exceptions = caught_exceptions if caught_exceptions is not None else {}
        self.event = threading.Event()

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


class Detector(Model):

    class Detection:
        def __init__(self, xmin, ymin, xmax, ymax, score, class_id):
            self.xmin = xmin
            self.xmax = xmax
            self.ymin = ymin
            self.ymax = ymax
            self.score = score
            self.class_id = class_id

    def __init__(self, *args, labels_map=None, keep_aspect_ratio_resize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_aspect_ratio_resize = keep_aspect_ratio_resize
        self.labels_map = labels_map

        self.image_blob_name, self.image_info_blob_name = self._get_inputs(self.net)
        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
        assert self.n == 1, 'Only batch size == 1 is supported.'

        self.output_parser = self._get_output_parser(self.net, self.image_blob_name)

    def _get_inputs(self, net):
        image_blob_name = None
        image_info_blob_name = None
        for blob_name, blob in net.input_info.items():
            if len(blob.input_data.shape) == 4:
                image_blob_name = blob_name
            elif len(blob.input_data.shape) == 2:
                image_info_blob_name = blob_name
            else:
                raise RuntimeError('Unsupported {}D input layer "{}". Only 2D and 4D input layers are supported'
                                   .format(len(blob.shape), blob_name))
        if image_blob_name is None:
            raise RuntimeError('Failed to identify the input for the image.')
        return image_blob_name, image_info_blob_name

    def _get_output_parser(self, net, image_blob_name, bboxes='bboxes', labels='labels', scores='scores'):
        try:
            parser = SingleOutputParser(net.outputs)
            log.info('Use SingleOutputParser')
            return parser
        except ValueError:
            pass

        try:
            parser = MultipleOutputParser(net.outputs, bboxes, scores, labels)
            log.info('Use MultipleOutputParser')
            return parser
        except ValueError:
            pass

        try:
            h, w = net.input_info[image_blob_name].input_data.shape[2:]
            parser = BoxesLabelsParser([w, h], net.outputs)
            log.info('Use BoxesLabelsParser')
            return parser
        except ValueError:
            pass
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
        for detection in detections:
            detection.xmin *= scale_x
            detection.xmax *= scale_x
            detection.ymin *= scale_y
            detection.ymax *= scale_y
        return detections


class SingleOutputParser:
    def __init__(self, all_outputs):
        if len(all_outputs) != 1:
            raise ValueError('Network must have only one output.')
        self.output_name, output_data = next(iter(all_outputs.items()))
        last_dim = np.shape(output_data)[-1]
        if last_dim != 7:
            raise ValueError('The last dimension of the output blob must be equal to 7, '
                             'got {} instead.'.format(last_dim))

    def __call__(self, outputs):
        return [Detector.Detection(xmin, ymin, xmax, ymax, score, label)
                for _, label, score, xmin, ymin, xmax, ymax in outputs[self.output_name][0][0]]


def find_layer_by_name(name, all_outputs):
    suitable_layers = [layer_name for layer_name in all_outputs if name in layer_name]
    if not suitable_layers:
        raise ValueError('Suitable layer for "{}" output is not found'.format(name))

    if len(suitable_layers) > 1:
        raise ValueError('More than 1 layer matched to "{}" output'.format(name))

    return suitable_layers[0]


class MultipleOutputParser:
    def __init__(self, all_outputs, bboxes_layer='bboxes', scores_layer='scores', labels_layer='labels'):
        self.labels_layer = find_layer_by_name(labels_layer, all_outputs)
        self.scores_layer = find_layer_by_name(scores_layer, all_outputs)
        self.bboxes_layer = find_layer_by_name(bboxes_layer, all_outputs)

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer][0]
        scores = outputs[self.scores_layer][0]
        labels = outputs[self.labels_layer][0]
        return [Detector.Detection(*bbox, score, label) for label, score, bbox in zip(labels, scores, bboxes)]


class BoxesLabelsParser:
    def __init__(self, input_size, all_outputs, labels_layer='labels', default_label=1):
        try:
            self.labels_layer = find_layer_by_name(labels_layer, all_outputs)
            log.info('Use output "{}" as the one containing labels of detected objects.'
                     .format(self.labels_layer))
        except ValueError:
            log.warning('Suitable layer for "{}" output is not found. '
                        'Treating detector as class agnostic with output label {}'
                        .format(labels_layer, default_label))
            self.labels_layer = None
            self.default_label = default_label

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
        
        detections = [Detector.Detection(*bbox, score, label) for label, score, bbox in zip(labels, scores, bboxes)]
        return detections


class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1


class ModeInfo:
    def __init__(self):
        self.last_start_time = perf_counter()
        self.last_end_time = None
        self.frames_count = 0
        self.latency_sum = 0


def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}
    config_min_latency = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
                           if num_streams.isdigit() \
                           else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                                                              if int(devices_nstreams['CPU']) > 0 \
                                                              else 'CPU_THROUGHPUT_AUTO'

        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                                                              if int(devices_nstreams['GPU']) > 0 \
                                                              else 'GPU_THROUGHPUT_AUTO'

        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'
    
    return config_user_specified, config_min_latency


def main():
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    config_user_specified, config_min_latency = get_plugin_configs(args.device, args.num_streams, args.num_threads)

    labels_map = None
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]

    log.info('Loading network...')
    completed_request_results = {}
    modes = cycle(Modes)
    prev_mode = mode = next(modes)
    log.info('Using {} mode'.format(mode.name))
    mode_info = {mode: ModeInfo()}
    exceptions = []

    detectors = {
        Modes.USER_SPECIFIED:
            Detector(ie, args.model, device=args.device, plugin_config=config_user_specified,
                     results=completed_request_results, max_num_requests=args.num_infer_requests,
                     labels_map=labels_map, keep_aspect_ratio_resize=args.keep_aspect_ratio,
                     caught_exceptions=exceptions),
        Modes.MIN_LATENCY:
            Detector(ie, args.model, device=args.device.split(':')[-1].split(',')[0], plugin_config=config_min_latency,
                     results=completed_request_results, max_num_requests=1,
                     labels_map=labels_map, keep_aspect_ratio_resize=args.keep_aspect_ratio,
                     caught_exceptions=exceptions)
    }

    try:
        input_stream = int(args.input)
    except ValueError:
        input_stream = args.input
    cap = cv2.VideoCapture(input_stream)
    wait_key_time = 1

    next_frame_id = 0
    next_frame_id_to_show = 0
    input_repeats = 0
  
    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between min_latency/user_specified modes, press TAB key in the output window")

    palette = ColorPalette(len(labels_map) if labels_map is not None else 100)
    presenter = monitors.Presenter(args.utilization_monitors, 55,
        (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))

    while (cap.isOpened() \
           or completed_request_results \
           or len(detectors[mode].empty_requests) < len(detectors[mode].requests)) \
          and not exceptions:
        if next_frame_id_to_show in completed_request_results:
            frame_meta, raw_outputs = completed_request_results.pop(next_frame_id_to_show)
            objects = detectors[mode].postprocess(raw_outputs, frame_meta)

            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(objects) and args.raw_output_message:
                log.info(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')

            origin_im_size = frame.shape[:-1]
            presenter.drawGraphs(frame)
            for obj in objects:
                if obj.score > args.prob_threshold:
                    xmin = max(int(obj.xmin), 0)
                    ymin = max(int(obj.ymin), 0)
                    xmax = min(int(obj.xmax), origin_im_size[1])
                    ymax = min(int(obj.ymax), origin_im_size[0])
                    class_id = int(obj.class_id)
                    color = palette[class_id]
                    det_label = labels_map[class_id] if labels_map and len(labels_map) >= class_id else str(class_id)

                    if args.raw_output_message:
                        log.info(
                            '{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '.format(det_label, obj.score,
                                                                                 xmin, ymin, xmax, ymax))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, '#{} {:.1%}'.format(det_label, obj.score),
                                (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            mode_message = '{} mode'.format(mode.name)
            put_highlighted_text(frame, mode_message, (10, int(origin_im_size[0] - 20)),
                                 cv2.FONT_HERSHEY_COMPLEX, 0.75, (10, 10, 200), 2)

            next_frame_id_to_show += 1
            if prev_mode == mode:
                mode_info[mode].frames_count += 1
            elif len(completed_request_results) == 0:
                mode_info[prev_mode].last_end_time = perf_counter()
                prev_mode = mode

            # Frames count is always zero if mode has just been switched (i.e. prev_mode != mode).
            if mode_info[mode].frames_count != 0:
                fps_message = 'FPS: {:.1f}'.format(mode_info[mode].frames_count / \
                                                   (perf_counter() - mode_info[mode].last_start_time))
                mode_info[mode].latency_sum += perf_counter() - start_time
                latency_message = 'Latency: {:.1f} ms'.format((mode_info[mode].latency_sum / \
                                                              mode_info[mode].frames_count) * 1e3)
                # Draw performance stats over frame.
                put_highlighted_text(frame, fps_message, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
                put_highlighted_text(frame, latency_message, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)

            if not args.no_show:
                cv2.imshow('Detection Results', frame)
                key = cv2.waitKey(wait_key_time)

                ESC_KEY = 27
                TAB_KEY = 9
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                # Switch mode.
                # Disable mode switch if the previous switch has not been finished yet.
                if key == TAB_KEY and mode_info[mode].frames_count > 0:
                    mode = next(modes)
                    detectors[prev_mode].await_all()
                    mode_info[prev_mode].last_end_time = perf_counter()
                    mode_info[mode] = ModeInfo()
                    log.info('Using {} mode'.format(mode.name))
                else:
                    presenter.handleKey(key)

        elif detectors[mode].empty_requests and cap.isOpened():
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                if input_repeats < args.loop_input or args.loop_input < 0:
                    cap.open(input_stream)
                    input_repeats += 1
                else:
                    cap.release()
                continue

            detectors[mode](frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            detectors[mode].await_any()

    if exceptions:
        raise exceptions[0]

    for exec_net in detectors.values():
        exec_net.await_all()

    for mode_value, mode_stats in mode_info.items():
        log.info('')
        log.info('Mode: {}'.format(mode_value.name))

        end_time = mode_stats.last_end_time if mode_stats.last_end_time is not None \
                                            else perf_counter()
        log.info('FPS: {:.1f}'.format(mode_stats.frames_count / \
                                      (end_time - mode_stats.last_start_time)))
        log.info('Latency: {:.1f} ms'.format((mode_stats.latency_sum / \
                                             mode_stats.frames_count) * 1e3))
    print(presenter.reportMeans())


if __name__ == '__main__':
    sys.exit(main() or 0)
