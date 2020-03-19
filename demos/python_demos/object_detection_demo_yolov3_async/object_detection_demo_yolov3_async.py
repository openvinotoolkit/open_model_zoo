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
from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time
from queue import Queue

import cv2
from openvino.inference_engine import IENetwork, IECore

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    args.add_argument("-nireq", "--num_infer_requests", help="Optional. Number of infer requests",
                      default=2, type=int)
    args.add_argument("-nstreams", "--num_streams", help="Optional. Number of streams to use for inference on "
                      "the CPU or/and GPU in throughput mode (for HETERO and MULTI device cases use format "
                      "<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)",
                      default="", type=str)
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                        help="Optional. Number of threads to use for inference on CPU (including HETERO cases)")
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')
    return parser


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def main():
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # -------------- 1. Load extensions library (if specified), configure numbers of threads and streams ---------------
    log.info("Creating Inference Engine...")
    ie_sync = IECore()
    ie_async = IECore()

    if args.cpu_extension and 'CPU' in args.device:
        ie_sync.add_extension(args.cpu_extension, 'CPU')
        ie_async.add_extension(args.cpu_extension, 'CPU')

    ie_sync.set_config({'CPU_THREADS_NUM': '1'}, 'CPU')
    if args.number_threads is not None:
        ie_async.set_config({'CPU_THREADS_NUM': str(args.number_threads)}, 'CPU')
    ie_async.set_config({'CPU_BIND_THREAD': 'NO'}, 'CPU')

    ie_sync.set_config({'CPU_THROUGHPUT_STREAMS': '1'}, 'CPU')
    ie_sync.set_config({'GPU_THROUGHPUT_STREAMS': '1'}, 'GPU')
    if args.num_streams:
        if args.num_streams.isdigit():
            devices_nstreams = {device: int(args.num_streams) for device in ['CPU', 'GPU']}
        else:
            devices_nstreams = {device.split(':')[0]: device.split(':')[1] for device in args.num_streams.split(',')}
        
        if 'CPU' in devices_nstreams:
            ie_async.set_config({'CPU_THROUGHPUT_STREAMS': str(devices_nstreams['CPU']) \
                                                           if int(devices_nstreams['CPU']) > 0 \
                                                           else 'CPU_THROUGHPUT_AUTO'}, 'CPU')

        if 'GPU' in devices_nstreams:
            ie_async.set_config({'GPU_THROUGHPUT_STREAMS': str(devices_nstreams['GPU']) \
                                                           if int(devices_nstreams['GPU']) > 0 \
                                                           else 'GPU_THROUGHPUT_AUTO'}, 'GPU')

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
    if "CPU" in args.device:
        supported_layers = ie_sync.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")
    input_blob = next(iter(net.inputs))

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    input_stream = 0 if args.input == "cam" else args.input

    is_async_mode = True
    cap = cv2.VideoCapture(input_stream)
    number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

    wait_key_code = 1

    # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
    if number_input_frames != 1:
        ret, frame = cap.read()
    else:
        is_async_mode = False
        wait_key_code = 0

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_net_sync = ie_sync.load_network(network=net, num_requests=1, device_name=args.device)
    exec_net_async = ie_async.load_network(network=net, num_requests=args.num_infer_requests, device_name=args.device)

    next_request_id = 0
    empty_request_ids = Queue(maxsize=args.num_infer_requests)
    for i in range(args.num_infer_requests):
        empty_request_ids.put(i)
    frames = [None] * args.num_infer_requests
    frame_buffer = Queue()
    raw_frame = None
    sync_frame = None
    render_time = 0
    parsing_time = 0
    new_shape = None

    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")

    def process_infer_request_output(output, frame, next_request_id):
        objects = list()
        render_time = 0

        start_time = time()
        for layer_name, out_blob in output.items():
            out_blob = out_blob.reshape(net.layers[net.layers[layer_name].parents[0]].out_data[0].shape)
            layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
            log.info("Layer {} parameters: ".format(layer_name))
            layer_params.log_params()
            objects += parse_yolo_region(out_blob, new_shape,
                                         raw_frame.shape[:-1], layer_params,
                                         args.prob_threshold)
        parsing_time = time() - start_time

        # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
                    objects[j]['confidence'] = 0

        # Drawing objects with respect to the --prob_threshold CLI parameter
        objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]

        if len(objects) and args.raw_output_message:
            log.info("\nDetected boxes for batch {}:".format(1))
            log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

        origin_im_size = frame.shape[:-1]
        for obj in objects:
            # Validation bbox of detected object
            if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] \
               or obj['xmin'] < 0 or obj['ymin'] < 0:
                continue
            color = (int(min(obj['class_id'] * 12.5, 255)),
                     min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
            det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                str(obj['class_id'])

            if args.raw_output_message:
                log.info(
                    "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'],
                                                                              obj['xmin'], obj['ymin'], obj['xmax'],
                                                                              obj['ymax'],
                                                                              color))

            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            cv2.putText(frame,
                        "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                        (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        # Draw performance stats over frame
        inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
            "Inference time: {:.3f} ms".format(det_time * 1e3)
        render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
        async_mode_message = "Async mode is on. Processing request {}".format(next_request_id) if is_async_mode \
           else "Async mode is off. Processing request {}".format(next_request_id)
        parsing_message = "YOLO parsing time is {:.3f} ms".format(parsing_time * 1e3)

        # Background for messages with metrics
        cv2.rectangle(frame, (10, 0), (335, 55), (0, 0, 0), -1)

        cv2.putText(frame, inf_time_message, (15, 15),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, render_time_message, (15, 45),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        cv2.putText(frame, async_mode_message, (10, int(origin_im_size[0] - 20)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        cv2.putText(frame, parsing_message, (15, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        start_time = time()
        if not args.no_show:
            cv2.imshow("DetectionResults", frame)
        render_time = time() - start_time

    while cap.isOpened():
        if is_async_mode:
            if exec_net_async.requests[next_request_id].wait(0) == 0 and not frames[next_request_id] is None:
                output = exec_net_async.requests[next_request_id].outputs

                process_infer_request_output(output, frames[next_request_id], next_request_id)
                frames[next_request_id] = None

                empty_request_ids.put(next_request_id)
                next_request_id += 1
                if next_request_id == args.num_infer_requests:
                    next_request_id = 0
            elif not empty_request_ids.empty():
                if frame_buffer.empty():
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    frame = frame_buffer.get()
                raw_frame = frame.copy()

                request_id = empty_request_ids.get()
                in_frame = cv2.resize(frame, (w, h))

                # resize input_frame to network size
                in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                in_frame = in_frame.reshape((n, c, h, w))
                new_shape = in_frame.shape[2:]
                frames[request_id] = raw_frame.copy()

                # Start inference
                start_time = time()
                exec_net_async.start_async(request_id=request_id, inputs={input_blob: in_frame})
                det_time = time() - start_time
        else:
            if exec_net_sync.requests[0].wait(0) == 0 and not sync_frame is None:
                output = exec_net_sync.requests[0].outputs

                process_infer_request_output(output, sync_frame, 0)
                sync_frame = None
            elif sync_frame is None:
                if frame_buffer.empty():
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    frame = frame_buffer.get()
                raw_frame = frame.copy()
                
                in_frame = cv2.resize(frame, (w, h))

                # resize input_frame to network size
                in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                in_frame = in_frame.reshape((n, c, h, w))
                new_shape = in_frame.shape[2:]
                sync_frame = raw_frame.copy()

                # Start inference
                start_time = time()
                exec_net_sync.start_async(request_id=0, inputs={input_blob: in_frame})
                det_time = time() - start_time

        if not args.no_show:
            key = cv2.waitKey(wait_key_code)

            # ESC key
            if key == 27:
                break
            # Tab key
            if key == 9:
                if is_async_mode:
                    for i in [*range(next_request_id, args.num_infer_requests), *range(next_request_id)]:
                        if not frames[i] is None:
                            frame_buffer.put(frames[i].copy())
                            frames[i] = None
                        else:
                            break
                            
                    exec_net_sync.requests[0].wait(-1)
                else:
                    sync_frame = None

                    for i in range(args.num_infer_requests):
                        exec_net_async.requests[i].wait(-1)
                        
                next_request_id = 0
                empty_request_ids = Queue(maxsize=args.num_infer_requests)
                for i in range(args.num_infer_requests):
                    empty_request_ids.put(i)
                frames = [None] * args.num_infer_requests
                raw_frame = None

                is_async_mode = not is_async_mode
                log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
