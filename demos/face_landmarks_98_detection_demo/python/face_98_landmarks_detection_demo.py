#!/usr/bin/env python3
"""
 Copyright (C) 2020-2021 Intel Corporation

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
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
import models
import monitors
from images_capture import open_images_capture
from pipelines import get_user_config, AsyncPipeline
from performance_metrics import PerformanceMetrics
from helpers import resolution
from ie_module import Module
from helpers import resolution

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m_lm', help='Required. Path to an .xml file with a 98 landmarks detection trained model.',
                      required=True, type=Path)
    args.add_argument('-m_fd', type=Path, required=True,
                        help='Required. Path to an .xml file with Face Detection model.')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of output to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                       help='Optional. Number of frames to store in output. '
                            'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    args.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
                        help='Optional. Specify the input size of detection model for '
                             'reshaping. Example: 500 700.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')
    infer_args.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.5,
                       help='Optional. Probability threshold for face detections.')
    infer_args.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help='Optional. Scaling ratio for bboxes passed to face recognition.')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-no_show', '--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser

def get_config(self, device):
    config = {
        "PERF_COUNT": "YES" if self.perf_count else "NO",
    }
    if device == 'GPU' and self.gpu_ext:
        config['CONFIG_FILE'] = self.gpu_ext
    return config


class LandmarksDetector(Module):

    def __init__(self, ie, model):
        super(LandmarksDetector, self).__init__(ie, model)

        assert len(self.net.input_info) == 1, 'Expected 1 input blob'
        assert len(self.net.outputs) == 1, 'Expected 1 output blob'
        self.input_blob = next(iter(self.net.input_info))
        self.output_blob = next(iter(self.net.outputs))
        self.input_shape = self.net.input_info[self.input_blob].input_data.shape
        output_shape = self.net.outputs[self.output_blob].shape


    def _xywh2cs(self, image_size, x, y, w, h, padding=1.25):
        aspect_ratio = image_size[0] / image_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
 
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding

        return center, scale


    def _flip_back(self, output_flipped):

        assert output_flipped.ndim == 4, \
            'output_flipped should be [batch_size, num_keypoints, height, width]'
        flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
                      [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
                      [12, 20], [13, 19], [14, 18], [15, 17], [33, 46], [34, 45],
                      [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48],
                      [41, 47], [60, 72], [61, 71], [62, 70], [63, 69], [64, 68],
                      [65, 75], [66, 74], [67, 73], [55, 59], [56, 58], [76, 82],
                      [77, 81], [78, 80], [87, 83], [86, 84], [88, 92],
                      [89, 91], [95, 93], [96, 97]]
        shape_ori = output_flipped.shape
        channels = 1
        output_flipped = output_flipped.reshape(shape_ori[0], -1, channels,
                                                shape_ori[2], shape_ori[3])
        output_flipped_back = output_flipped.copy()

        # Swap left-right parts
        for left, right in flip_pairs:
            output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
            output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
        output_flipped_back = output_flipped_back.reshape(shape_ori)
        # Flip horizontally
        output_flipped_back = output_flipped_back[..., ::-1]
        return output_flipped_back


    def _keypoints_from_heatmaps(self, heatmaps, center, scale):

        def _get_max_preds(heatmaps):
            assert isinstance(heatmaps,
                              np.ndarray), ('heatmaps should be numpy.ndarray')
            assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

            N, K, _, W = heatmaps.shape
            heatmaps_reshaped = heatmaps.reshape((N, K, -1))
            idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
            maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

            preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
            preds[:, :, 0] = preds[:, :, 0] % W
            preds[:, :, 1] = preds[:, :, 1] // W

            preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
            return preds, maxvals

        def _get_3rd_point(a, b):
            direction = a - b
            third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)
            return third_pt

        def rotate_point(pt, angle_rad):
            sn, cs = np.sin(angle_rad), np.cos(angle_rad)
            new_x = pt[0] * cs - pt[1] * sn
            new_y = pt[0] * sn + pt[1] * cs
            rotated_pt = [new_x, new_y]

            return rotated_pt

        def _get_affine_transform(center, scale, rot, output_size, shift=(0., 0.), inv=False):

            scale_tmp = scale * 200.0

            shift = np.array(shift)
            src_w = scale_tmp[0]
            dst_w = output_size[0]
            dst_h = output_size[1]

            rot_rad = np.pi * rot / 180
            src_dir = rotate_point([0., src_w * -0.5], rot_rad)
            dst_dir = np.array([0., dst_w * -0.5])

            src = np.zeros((3, 2), dtype=np.float32)
            src[0, :] = center + scale_tmp * shift
            src[1, :] = center + src_dir + scale_tmp * shift
            src[2, :] = _get_3rd_point(src[0, :], src[1, :])

            dst = np.zeros((3, 2), dtype=np.float32)
            dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
            dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
            dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

            if inv:
                trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
            else:
                trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

            return trans

        def _transform_preds(coords, center, scale, output_size, use_udp=False):
            target_coords = coords.copy()
            trans = _get_affine_transform(center, scale, 0, output_size, inv=True)
            for p in range(coords.shape[0]):
                target_coords[p, 0:2] = np.array(trans) @ np.array([coords[p, 0:2][0], coords[p, 0:2][1], 1.])
            return target_coords

        N, K, H, W = heatmaps.shape

        preds, maxvals = _get_max_preds(heatmaps)        

        for n in range(N):
            for k in range(K):
                heatmap = heatmaps[n][k]
                px = int(preds[n][k][0])
                py = int(preds[n][k][1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    preds[n][k] += np.sign(diff) * .25

        # Transform back to the image
        for i in range(N):
            preds[i] = _transform_preds(
                preds[i], center, scale, [W, H])

        return preds, maxvals 

    def preprocess(self, inputs):
        def cut_rois(inputs):
            frame = inputs["frame"].copy()
            roi = inputs["roi"]
            inputs = []           
            inputs.append(frame[roi[2]:roi[3], roi[0]:roi[1], :])
            return inputs

        def resize_input(frame, shape):            
            return cv2.resize(frame, (shape[2], shape[3]))    
        roi = inputs["roi"]
        shape = inputs["shape"]
        inputs = cut_rois(inputs)
        inputs = [resize_input(input, self.input_shape).transpose(2,0,1) for input in inputs]
        return {"image" : inputs[0]}, {"roi" : roi, "shape" : shape}

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_blob: input})

    def start_async(self, inputs):
        inputs = self.preprocess(inputs)
        for input in inputs:
            self.enqueue(input)

    def postprocess(self, raw_result, preprocess_meta):
        roi = preprocess_meta["roi"]
        raw_result = raw_result['3148']
        raw_result[:, :, :, 1:] = raw_result[:, :, :, :-1]
        center, scale = self._xywh2cs(preprocess_meta["shape"], roi[0], roi[2], roi[1] - roi[0], roi[3] - roi[2] )
        landmarks, maxvals = self._keypoints_from_heatmaps(raw_result, center, scale)
        return landmarks


def get_rois(frame, detections, threshold):
    size = frame.shape[:2]
    rois = []
    for detection in detections:
        if detection.score > threshold:
            class_id = int(detection.id)
            w = detection.xmax - detection.xmin
            h = detection.ymax - detection.ymin
            xmin = max(int(detection.xmin - w*0.2), 0)
            ymin = max(int(detection.ymin - h*0.2), 0)
            xmax = min(int(detection.xmax + w*0.2), size[1])
            ymax = min(int(detection.ymax + h*0.2), size[0])
            rois.append([xmin, xmax, ymin, ymax])
    return rois

def main():
    args = build_argparser().parse_args()
    metrics = PerformanceMetrics()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)

    cap = open_images_capture(args.input, args.loop)

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    log.info('Loading face detection network...')

    input_transform = models.InputTransform(False, None, None)
    common_args = (ie, args.m_fd, input_transform)
    fd_model = models.SSD(*common_args,labels=["face"], keep_aspect_ratio_resize=False)
    fd_pipeline = AsyncPipeline(ie, fd_model, plugin_config, device=args.device, max_num_requests=args.num_infer_requests)
    log.info('The model {} is loaded to {}'.format(args.m_fd, args.device))

    log.info('Loading landmarks detection network...')
    lms_model = LandmarksDetector(ie, args.m_lm)
    lms_pipeline = AsyncPipeline(ie, lms_model, plugin_config, device=args.device, max_num_requests=args.num_infer_requests)

    log.info('Starting inference...')

    next_frame_id = 0
    next_frame_id_to_process = 0
    next_frame_id_lms = 0
    next_frame_id_lms_by_frame = []
    output_resolution = 0
    output_transform = None

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    
    while True:     
        if fd_pipeline.callback_exceptions:
            raise fd_pipeline.callback_exceptions[0]   
        detections = fd_pipeline.get_result(next_frame_id_to_process)

        if detections:
            objects, frame_meta = detections
            frame2show = frame_meta['frame'].copy()
            rois = get_rois(frame2show, objects, args.t_fd)
            for roi in rois:
                inputs = {"frame" : frame2show, "roi" : roi, "shape" : frame2show.shape[:2]}
                if lms_pipeline.is_ready():
                    start_time = perf_counter()
                    lms_pipeline.submit_data(inputs, next_frame_id_lms, {'frame': frame2show, 'start_time': start_time})
                    next_frame_id_lms_by_frame.append(next_frame_id_lms)
                    next_frame_id_lms += 1                
            next_frame_id_to_process += 1
            lms_pipeline.await_all()
            while len(next_frame_id_lms_by_frame) > 0:
                for frame_id in next_frame_id_lms_by_frame:
                    landmarks = lms_pipeline.get_result(frame_id)
                    if landmarks:
                        for p1,p2 in landmarks[0][0]:
                            cv2.circle(frame2show,(int(p1),int(p2)),1,(0,0,255),2)
                        next_frame_id_lms_by_frame.remove(frame_id)
            
            cv2.imshow("face_landmarks_98_demo", frame2show)
            cv2.waitKey(1)

        if fd_pipeline.is_ready():
            if next_frame_id == 0:
                output_transform = models.OutputTransform(frame.shape[:2], args.output_resolution)
                if args.output_resolution:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (frame.shape[1], frame.shape[0])
                presenter = monitors.Presenter(args.utilization_monitors, 55,
                                               (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                         cap.fps(), output_resolution):
                    raise RuntimeError("Can't open video writer")

            video_writer = cv2.VideoWriter()
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(),
                output_resolution):
                raise RuntimeError("Can't open video writer")

            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                break
            # Submit for inference
            fd_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            fd_pipeline.await_any()
                

if __name__ == '__main__':
    sys.exit(main() or 0)


