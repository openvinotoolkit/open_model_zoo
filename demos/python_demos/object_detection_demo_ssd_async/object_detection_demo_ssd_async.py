#!/usr/bin/env python3
"""
 Copyright (C) 2018-2019 Intel Corporation

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

import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log

from openvino.inference_engine import IECore

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))
import monitors


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')
    args.add_argument("-u", "--utilization_monitors", default='', type=str,
                      help="Optional. List of monitors to show initially.")

    return parser


class SingleOutputPostprocessor:
    def __init__(self, output_layer):
        self.output_layer = output_layer

    def __call__(self, outputs):
        return outputs[self.output_layer].buffer[0][0]


class MultipleOutputPostprocessor:
    def __init__(self, bboxes_layer='bboxes', scores_layer='scores', labels_layer='labels'):
        self.bboxes_layer = bboxes_layer
        self.scores_layer = scores_layer
        self.labels_layer = labels_layer

    def __call__(self, outputs):
        bboxes = outputs[self.bboxes_layer].buffer[0]
        scores = outputs[self.scores_layer].buffer[0]
        labels = outputs[self.labels_layer].buffer[0]
        return [[0, label, score, *bbox] for label, score, bbox in zip(labels, scores, bboxes)]


def get_output_postprocessor(net, bboxes='bboxes', labels='labels', scores='scores'):
    if len(net.outputs) == 1:
        output_blob = next(iter(net.outputs))
        return SingleOutputPostprocessor(output_blob)
    elif len(net.outputs) >= 3:
        def find_layer(name, all_outputs):
            suitable_layers = [layer_name for layer_name in all_outputs if name in layer_name]
            if not suitable_layers:
                raise ValueError('Suitable layer for "{}" output is not found'.format(name))

            if len(suitable_layers) > 1:
                raise ValueError('More than 1 layer matched to "{}" output'.format(name))

            return suitable_layers[0]

        labels_out = find_layer(labels, net.outputs)
        scores_out = find_layer(scores, net.outputs)
        bboxes_out = find_layer(bboxes, net.outputs)

        return MultipleOutputPostprocessor(bboxes_out, scores_out, labels_out)

    raise RuntimeError("Unsupported model outputs")


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network")
    net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.input_info:
        if len(net.input_info[blob_name].input_data.shape) == 4:
            input_blob = blob_name
        elif len(net.input_info[blob_name].input_data.shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.input_info[blob_name].input_data.shape), blob_name))

    output_postprocessor = get_output_postprocessor(net)

    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    # Read and pre-process input image
    n, c, h, w = net.input_info[input_blob].input_data.shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + str(input_stream)

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...")
    is_async_mode = True
    render_time = 0
    if is_async_mode:
        ret, frame = cap.read()
        frame_h, frame_w = frame.shape[:2]

    presenter = monitors.Presenter(args.utilization_monitors, 45,
        (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")

    while cap.isOpened():
        if is_async_mode:
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
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            for obj in output_postprocessor(exec_net.requests[cur_request_id].output_blobs):
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    det_label = labels_map[class_id] if labels_map else str(class_id)
                    cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            # Draw performance stats
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1000)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                "Async mode is off. Processing request {}".format(cur_request_id)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(frame_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)

        presenter.drawGraphs(frame)
        render_start = time.time()
        if not args.no_show:
            cv2.imshow("Detection Results", frame)
        render_end = time.time()
        render_time = render_end - render_start

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
            frame_h, frame_w = frame.shape[:2]

        if not args.no_show:
            key = cv2.waitKey(1)
            if key == 27:
                break
            if (9 == key):
                is_async_mode = not is_async_mode
                log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
            else:
                presenter.handleKey(key)
    print(presenter.reportMeans())


if __name__ == '__main__':
    sys.exit(main() or 0)
