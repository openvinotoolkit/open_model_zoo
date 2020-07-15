#!/usr/bin/env python
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

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
import collections
import numpy as np

from openvino.inference_engine import IECore


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

    args.add_argument("--num_requests", type=int, help="Use 1 for Sync Mode."
                                                        "Use value of 2 or more for Async.",
                                                        default=2,
                                                        required=True
                                                        )

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    
    # Inference mode decided on the basis of num_requests parameter     
    if args.num_requests == 0:
    	log.error("Invalid request number, try again with an integer greater than equal to 1")
    	sys.exit(1)
    if args.num_requests == 1:
        is_async_mode = False
        log.info("Starting inference in sync mode...")
    elif args.num_requests > 1:
        is_async_mode = True
        log.info("Starting inference in Async mode...")


    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    
    log.info("Creating Inference Engine...")
    ie = IECore()

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
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=args.num_requests, device_name=args.device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + input_stream

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None


    # rotate requests instead of swapping
    # with this, more than 2 requests can be started
    request_id_list = list(range(args.num_requests))
    rotate_request_by = len(request_id_list)-1
    current_request_id_list = collections.deque(request_id_list)
    predict_request_id_list = current_request_id_list.copy()
    predict_request_id_list.rotate(rotate_request_by)
    frame_list = [None]*args.num_requests

    
    render_time = 0

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
 
    FPS = []
    res = None
    frame = None

    while cap.isOpened():

        ret, raw_frame = cap.read()

        if not ret:
            break

        raw_frame_h, raw_frame_w = raw_frame.shape[:2]

        in_frame = cv2.resize(raw_frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        feed_dict[input_blob] = in_frame



        inf_start = time.time()

        exec_net.start_async(request_id=predict_request_id_list[0], inputs=feed_dict)
        frame_list[predict_request_id_list[0]] = raw_frame
        predict_request_id_list.rotate(rotate_request_by)

        if exec_net.requests[current_request_id_list[0]].wait(-1) == 0:
            # Parse detection results of the current request
            res = exec_net.requests[current_request_id_list[0]].outputs[out_blob]
            frame = frame_list[current_request_id_list[0]]
        
        current_request_id_list.rotate(rotate_request_by)

        if res is None or frame is None:
            continue

        inf_end = time.time()
        det_time = inf_end - inf_start
        FPS = FPS[-100:]
        FPS.append(1/det_time)
        

        for obj in res[0][0]:
            # Draw only objects when probability more than specified threshold
            if obj[2] > args.prob_threshold:
                xmin = int(obj[3] * raw_frame_w)
                ymin = int(obj[4] * raw_frame_h)
                xmax = int(obj[5] * raw_frame_w)
                ymax = int(obj[6] * raw_frame_h)
                class_id = int(obj[1])
                # Draw box and label\class_id
                color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                det_label = labels_map[class_id] if labels_map else str(class_id)
                cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        # Draw performance stats
        inf_time_message = "Inference time: {:.3f}, Mean FPS: {} for async mode".format(det_time * 1000, int(np.mean(FPS))) \
        if is_async_mode else \
                "Inference time: {:.3f}, Mean FPS: {} for sync mode".format(det_time * 1000, int(np.mean(FPS)))
        render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
        async_mode_message = "Async mode is on. Processing request {}".format(current_request_id_list[0]) if is_async_mode else \
                "Async mode is off. Processing request {}".format(current_request_id_list[0])

        cv2.putText(frame, inf_time_message, (15, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        cv2.putText(frame, async_mode_message, (10, int(raw_frame_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)

        
        render_start = time.time()
        if not args.no_show:
            cv2.imshow("Detection Results", frame)
        render_end = time.time()
        render_time = render_end - render_start

        if not args.no_show:
            key = cv2.waitKey(1)
            if key == 27:
                break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    sys.exit(main() or 0)