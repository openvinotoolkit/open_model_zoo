#!/usr/bin/env python3
"""
 Copyright (c) 2020-2024 Intel Corporation
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

import argparse
import cv2
import logging as log
import numpy as np
from time import perf_counter
import sys
from pathlib import Path

from openvino import Core, get_version

from utils.network_wrappers import MaskRCNN, SemanticSegmentation
from utils.misc import MouseClick, check_pressed_keys

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

import monitors
from images_capture import open_images_capture
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

WINNAME = 'Whiteboard_inpainting_demo'


def expand_mask(detection, d):
    for i in range(len(detection[0])):
        detection[0][i][2] = extend_mask(detection[0][i][2], d)


def extend_mask(mask, d=70):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        for c in contour:
            for x, y in c:
                cv2.circle(mask, (x, y), d, (1, 1, 1), -1)
    return mask


def remove_background(img, kernel_size=(7, 7), blur_kernel_size=21, invert_colors=True):
    bgr_planes = cv2.split(img)
    result = []
    kernel = np.ones(kernel_size, np.uint8)
    for plane in bgr_planes:
        dilated_img = cv2.morphologyEx(plane, cv2.MORPH_OPEN, kernel)
        dilated_img = cv2.dilate(dilated_img, kernel)
        bg_img = cv2.medianBlur(dilated_img, blur_kernel_size)
        if invert_colors:
            diff_img = 255 - cv2.absdiff(plane, bg_img)
        else:
            diff_img = cv2.absdiff(plane, bg_img)
        result.append(diff_img)
    return cv2.merge(result)


def main():
    parser = argparse.ArgumentParser(description='Whiteboard inpainting demo')
    parser.add_argument('-i', '--input', required=True,
                         help='Required. Path to a video file or a device node of a web-camera.')
    parser.add_argument('--loop', default=False, action='store_true',
                        help='Optional. Enable reading the input in a loop.')
    parser.add_argument('-o', '--output', required=False,
                        help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    parser.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                        help='Optional. Number of frames to store in output. '
                             'If 0 is set, all frames are stored.')
    parser.add_argument('-m_i', '--m_instance_segmentation', type=str, required=False,
                        help='Required. Path to the instance segmentation model.')
    parser.add_argument('-m_s', '--m_semantic_segmentation', type=str, required=False,
                        help='Required. Path to the semantic segmentation model.')
    parser.add_argument('-t', '--threshold', type=float, default=0.6,
                        help='Optional. Threshold for person instance segmentation model.')
    parser.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    parser.add_argument('-d', '--device', type=str, default='CPU',
                        help='Optional. Specify a target device to infer on. CPU or GPU is '
                             'acceptable. The demo will look for a suitable plugin for the device specified.')
    parser.add_argument('-u', '--utilization_monitors', default='', type=str,
                        help='Optional. List of monitors to show initially.')
    args = parser.parse_args()

    cap = open_images_capture(args.input, args.loop)
    if cap.get_type() not in ('VIDEO', 'CAMERA'):
        raise RuntimeError("The input should be a video file or a numeric camera ID")

    if bool(args.m_instance_segmentation) == bool(args.m_semantic_segmentation):
        raise ValueError('Set up exactly one of segmentation models: '
                         '--m_instance_segmentation or --m_semantic_segmentation')

    labels_dir = Path(__file__).resolve().parents[3] / 'data/dataset_classes'
    mouse = MouseClick()
    if not args.no_show:
        cv2.namedWindow(WINNAME)
        cv2.setMouseCallback(WINNAME, mouse.get_points)

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    model_path = args.m_instance_segmentation if args.m_instance_segmentation else args.m_semantic_segmentation
    log.info('Reading model {}'.format(model_path))
    if args.m_instance_segmentation:
        labels_file = str(labels_dir / 'coco_80cl_bkgr.txt')
        segmentation = MaskRCNN(core, args.m_instance_segmentation, labels_file,
                                args.threshold, args.device)
    elif args.m_semantic_segmentation:
        labels_file = str(labels_dir / 'cityscapes_19cl_bkgr.txt')
        segmentation = SemanticSegmentation(core, args.m_semantic_segmentation, labels_file,
                                            args.threshold, args.device)
    log.info('The model {} is loaded to {}'.format(model_path, args.device))

    metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()
    black_board = False
    frame_number = 0
    key = -1

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    out_frame_size = (frame.shape[1], frame.shape[0] * 2)
    output_frame = np.full((frame.shape[0], frame.shape[1], 3), 255, dtype='uint8')
    presenter = monitors.Presenter(args.utilization_monitors, 20,
                                   (out_frame_size[0] // 4, out_frame_size[1] // 16))
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), out_frame_size):
        raise RuntimeError("Can't open video writer")

    while frame is not None:
        mask = None
        detections = segmentation.get_detections([frame])
        expand_mask(detections, frame.shape[1] // 27)
        if len(detections[0]) > 0:
            mask = detections[0][0][2]
            for i in range(1, len(detections[0])):
                mask = cv2.bitwise_or(mask, detections[0][i][2])

        if mask is not None:
            mask = np.stack([mask, mask, mask], axis=-1)
        else:
            mask = np.zeros(frame.shape, dtype='uint8')

        clear_frame = remove_background(frame, invert_colors=not black_board)

        output_frame = np.where(mask, output_frame, clear_frame)
        merged_frame = np.vstack([frame, output_frame])
        merged_frame = cv2.resize(merged_frame, out_frame_size)

        metrics.update(start_time, merged_frame)

        if video_writer.isOpened() and (args.output_limit <= 0 or frame_number <= args.output_limit-1):
            video_writer.write(merged_frame)

        presenter.drawGraphs(merged_frame)
        if not args.no_show:
            cv2.imshow(WINNAME, merged_frame)
            key = check_pressed_keys(key)
            if key == 27:  # 'Esc'
                break
            if key == ord('i'):  # catch pressing of key 'i'
                black_board = not black_board
                output_frame = 255 - output_frame
            else:
                presenter.handleKey(key)

        if mouse.crop_available:
            x0, x1 = min(mouse.points[0][0], mouse.points[1][0]), \
                     max(mouse.points[0][0], mouse.points[1][0])
            y0, y1 = min(mouse.points[0][1], mouse.points[1][1]), \
                     max(mouse.points[0][1], mouse.points[1][1])
            x1, y1 = min(x1, output_frame.shape[1] - 1), min(y1, output_frame.shape[0] - 1)
            board = output_frame[y0: y1, x0: x1, :]
            if board.shape[0] > 0 and board.shape[1] > 0:
                cv2.namedWindow('Board', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Board', board)

        frame_number += 1
        start_time = perf_counter()
        frame = cap.read()

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
