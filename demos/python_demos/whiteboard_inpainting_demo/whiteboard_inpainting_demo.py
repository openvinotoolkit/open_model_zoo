#!/usr/bin/env python3
"""
 Copyright (c) 2020 Intel Corporation
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
import time
import sys
from os import path as osp

from openvino.inference_engine import IECore  # pylint: disable=import-error,E0611

from utils.capture import VideoCapture
from utils.network_wrappers import MaskRCNN, SemanticSegmentation
from utils.misc import MouseClick, set_log_config, check_pressed_keys

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'common'))
import monitors

set_log_config()
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
    parser.add_argument('-i', type=str, help='Input sources (index of camera \
                        or path to a video file)', required=True)
    parser.add_argument('-m_i', '--m_instance_segmentation', type=str, required=False,
                        help='Path to the instance segmentation model')
    parser.add_argument('-m_s', '--m_semantic_segmentation', type=str, required=False,
                        help='Path to the semantic segmentation model')
    parser.add_argument('-t', '--threshold', type=float, default=0.6,
                        help='Threshold for person instance segmentation model')
    parser.add_argument('--output_video', type=str, default='', required=False,
                        help='Optional. Path to output video')
    parser.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    parser.add_argument('-d', '--device', type=str, default='CPU',
                        help='Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is '
                             'acceptable. The demo will look for a suitable plugin for the device specified')
    parser.add_argument('-l', '--cpu_extension', type=str, default=None,
                        help='MKLDNN (CPU)-targeted custom layers.Absolute \
                              path to a shared library with the kernels impl.')
    parser.add_argument('-u', '--utilization_monitors', default='', type=str,
                        help='Optional. List of monitors to show initially')
    args = parser.parse_args()

    capture = VideoCapture(args.i)

    if bool(args.m_instance_segmentation) == bool(args.m_semantic_segmentation):
        raise ValueError('Set up exactly one of segmentation models: '\
                         '--m_instance_segmentation or --m_semantic_segmentation')

    frame_size, fps = capture.get_source_parameters()
    out_frame_size = (int(frame_size[0]), int(frame_size[1] * 2))
    presenter = monitors.Presenter(args.utilization_monitors, 20,
                                   (out_frame_size[0] // 4, out_frame_size[1] // 16))

    root_dir = osp.dirname(osp.abspath(__file__))

    mouse = MouseClick()
    if not args.no_show:
        cv2.namedWindow(WINNAME)
        cv2.setMouseCallback(WINNAME, mouse.get_points)

    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(args.output_video, fourcc, fps, out_frame_size)
    else:
        output_video = None

    log.info("Initializing Inference Engine")
    ie = IECore()
    if args.m_instance_segmentation:
        labels_file = osp.join(root_dir, 'coco_labels.txt')
        segmentation = MaskRCNN(ie, args.m_instance_segmentation, labels_file,
                                args.threshold, args.device, args.cpu_extension)
    elif args.m_semantic_segmentation:
        labels_file = osp.join(root_dir, 'cityscapes_labels.txt')
        segmentation = SemanticSegmentation(ie, args.m_semantic_segmentation, labels_file,
                                            args.threshold, args.device, args.cpu_extension)

    black_board = False
    output_frame = np.full((frame_size[1], frame_size[0], 3), 255, dtype='uint8')
    frame_number = 0
    key = -1

    while True:
        start = time.time()
        _, frame = capture.get_frame()

        mask = None
        if frame is not None:
            detections = segmentation.get_detections([frame])
            expand_mask(detections, frame_size[0] // 27)
            if len(detections[0]) > 0:
                mask = detections[0][0][2]
                for i in range(1, len(detections[0])):
                    mask = cv2.bitwise_or(mask, detections[0][i][2])
        else:
            break

        if mask is not None:
            mask = np.stack([mask, mask, mask], axis=-1)
        else:
            mask = np.zeros(frame.shape, dtype='uint8')

        clear_frame = remove_background(frame, invert_colors=not black_board)

        output_frame = np.where(mask, output_frame, clear_frame)
        merged_frame = np.vstack([frame, output_frame])
        merged_frame = cv2.resize(merged_frame, out_frame_size)

        if output_video is not None:
            output_video.write(merged_frame)
        
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
            
        end = time.time()
        print('\rProcessing frame: {}, fps = {:.3}' \
            .format(frame_number, 1. / (end - start)), end="")
        frame_number += 1
    print('')

    log.info(presenter.reportMeans())
    
    if output_video is not None:
        output_video.release()
        

if __name__ == '__main__':
    main()
