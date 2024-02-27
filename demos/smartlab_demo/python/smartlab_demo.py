#!/usr/bin/env python3

"""
 Copyright (c) 2021-2024 Intel Corporation

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

import cv2
import time
import concurrent.futures

from display import Display
from evaluator import Evaluator
from openvino import Core
from argparse import ArgumentParser, SUPPRESS
from segmentor import Segmentor, SegmentorMstcn
from object_detection.detector import Detector


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-d", "--device", type=str, default='CPU', required=False,
                      help="Optional. Specify the target to infer on CPU or GPU.")
    args.add_argument('-tv', '--topview', required=True,
                      help='Required. Topview stream to be processed. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-sv', '--sideview', required=True,
                      help='Required. SideView to be processed. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-m_ta', '--m_topall', help='Required. Path to topview all class model.',
                      required=True, type=str)
    args.add_argument('-m_tm', '--m_topmove', help='Required. Path to topview moving class model.',
                      required=True, type=str)
    args.add_argument('-m_sa', '--m_sideall', help='Required. Path to sidetview all class model.',
                      required=True, type=str)
    args.add_argument('-m_sm', '--m_sidemove', help='Required. Path to sidetview moving class model.',
                      required=True, type=str)
    args.add_argument('--mode', default='multiview', help='Optional. Action recognition mode: multiview or mstcn',
                      type=str)
    args.add_argument('-m_en', '--m_encoder', help='Required for mstcn mode. Path to encoder model.',
                      required=False, type=str)
    args.add_argument('-m_en_t', '--m_encoder_top', help='Required for multiview mode. Path to encoder model for top view.',
                      required=False, type=str)
    args.add_argument('-m_en_s', '--m_encoder_side', help='Required for multiview mode. Path to encoder model for side view.',
                      required=False, type=str)
    args.add_argument('-m_de', '--m_decoder', help='Required. Path to decoder model.',
                      required=True, type=str)
    args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')

    return parser


def video_loop(args, cap_top, cap_side, detector, segmentor, evaluator, display):
    start_time = time.perf_counter()
    frame_counter = 0  # Frame index counter
    fps = 0.0
    interval_second = 1
    interval_start_frame = 0
    detector_result = None
    seg_results = None
    # multithread setup
    executor = concurrent.futures.ThreadPoolExecutor()
    future_detector = None
    buffer_display = None
    scoring = None
    keyframe = None

    while cap_top.isOpened() and cap_side.isOpened():
        ret_top, frame_top = cap_top.read()
        ret_side, frame_side = cap_side.read()
        if not ret_top or not ret_side:
            state = 'Finish'
            action_seg_results, top_det_results, side_det_results, frame_top, frame_side, frame_counter = buffer_display
            if frame_counter > 96:
                if seg_results is not None:
                    if args.mode == "mstcn":
                        action_seg_results = seg_results[-1]
                    # mode==multiview
                    else:
                        action_seg_results = seg_results

                display.display_result(
                    frame_top=frame_top,
                    frame_side=frame_side,
                    side_seg_results=action_seg_results,
                    top_seg_results=action_seg_results,
                    top_det_results=top_det_results,
                    side_det_results=side_det_results,
                    scoring=scoring,
                    state=state,
                    keyframe=keyframe,
                    frame_counter=frame_counter,
                    fps=fps)
            break
        else:
            if frame_counter % 10 == 0 and future_detector is None:
                future_detector = executor.submit(
                    detector.inference_multithread, frame_top, frame_side, frame_counter)
            seg_results = segmentor.inference(
                frame_top, frame_side, frame_counter)
            if seg_results is not None:
                if args.mode == 'mstcn':
                    display_seg_result = seg_results[-1]
                    seg_results = seg_results[:-1]
                else:
                    seg_results = seg_results[0]

            # get obj result
            if future_detector is not None and future_detector.done():
                detector_result = future_detector.result()
                future_detector = None

            current_time = time.perf_counter()
            current_frame = frame_counter
            if current_time - start_time > interval_second:
                total_frame_processed_in_interval = current_frame - interval_start_frame
                fps = total_frame_processed_in_interval / (current_time - start_time)
                interval_start_frame = current_frame
                start_time = current_time

            ''' The score evaluation module need to merge the results of the two modules and generate the scores '''
            if detector_result is not None:
                top_det_results, side_det_results = detector_result[0], detector_result[1]
                state, scoring, keyframe, action_seg_results = evaluator.inference(
                    top_det_results=top_det_results,
                    side_det_results=side_det_results,
                    action_seg_results=seg_results,
                    frame_top=frame_top,
                    frame_side=frame_side,
                    frame_counter=frame_counter,
                    mode=args.mode)
                buffer_display = action_seg_results, top_det_results, side_det_results, frame_top, frame_side, frame_counter
                if frame_counter >= 96:
                    if args.mode == 'mstcn':
                        action_seg_results = display_seg_result
                    display.display_result(
                        frame_top=frame_top,
                        frame_side=frame_side,
                        side_seg_results=action_seg_results,
                        top_seg_results=action_seg_results,
                        top_det_results=top_det_results,
                        side_det_results=side_det_results,
                        scoring=scoring,
                        state=state,
                        keyframe=keyframe,
                        frame_counter=frame_counter,
                        fps=fps)

        frame_counter += 1

        if cv2.waitKey(1) in {ord('q'), ord('Q'), 27}:  # Esc
            break


def main():
    args = build_argparser().parse_args()
    core = Core()

    ''' Object Detection Variables '''
    detector = Detector(
        core,
        args.device,
        [args.m_topall, args.m_topmove],
        [args.m_sideall, args.m_sidemove])

    '''Video Segmentation Variables'''
    if (args.mode == "multiview"):
        segmentor = Segmentor(core, args.device, args.m_encoder_side, args.m_encoder_top, args.m_decoder)
    elif (args.mode == "mtcnn"):
        segmentor = SegmentorMstcn(core, args.device, args.m_encoder, args.m_decoder)
    else:
        raise ValueError(f"Not supported mode: {args.sideview}")

    '''Score Evaluation Variables'''
    evaluator = Evaluator()

    '''Display Obj Detection, Action Segmentation and Score Evaluation Result'''
    display = Display(not args.no_show)

    """
        Process the video.
    """
    cap_top = cv2.VideoCapture(args.topview)
    if not cap_top.isOpened():
        raise ValueError(f"Can't open a video or frame from {args.topview}")
    cap_side = cv2.VideoCapture(args.sideview)
    if not cap_side.isOpened():
        raise ValueError(f"Can't open a video or frame from {args.sideview}")

    video_loop(
        args, cap_top, cap_side, detector, segmentor, evaluator, display)


if __name__ == "__main__":
    main()
