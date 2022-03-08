"""
 Copyright (C) 2021-2022 Intel Corporation

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
from collections import deque
from argparse import ArgumentParser, SUPPRESS

from display import Display
from evaluator import Evaluator
from openvino.runtime import Core
from segmentor import Segmentor, SegmentorMstcn
from object_detection.detector import Detector
from thread_argument import ThreadWithReturnValue




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
    args.add_argument('-m_ta', '--m_topall', help='Required. Path to topview all class model.', required=True, type=str)
    args.add_argument('-m_tm', '--m_topmove', help='Required. Path to topview moving class model.', required=True, type=str)
    args.add_argument('-m_sa', '--m_sideall', help='Required. Path to sidetview all class model.', required=True, type=str)
    args.add_argument('-m_sm', '--m_sidemove', help='Required. Path to sidetview moving class model.', required=True, type=str)
    args.add_argument('--mode', default='multiview', help='Optional. action recognition mode: multiview or mstcn', type=str)
    args.add_argument('-m_en', '--m_encoder', help='Required. Path to encoder model.', required=True, type=str)
    args.add_argument('-m_de', '--m_decoder', help='Required. Path to decoder model.', required=True, type=str)

    return parser

def video_loop(args, cap_top, cap_side, detector, segmentor, evaluator, display, buffer1, buffer2):
    old_time = time.time()
    frame_counter = 0 # Frame index counter
    fps = 0.0
    interval_second = 1
    interval_start_frame = 0
    total_frame_processed_in_interval = 0.0
    detector_result = None
    segmentor_result = None

    executor  = concurrent.futures.ThreadPoolExecutor()
    future_detector = None
    future_segmentor = None

    while cap_top.isOpened() and cap_side.isOpened():
        ret_top, frame_top = cap_top.read()  # frame:480 x 640 x 3
        ret_side, frame_side = cap_side.read()

        frame_counter += 1
        if not ret_top or not ret_side:
            break
        else:
            # dector with new thread
            if frame_counter % 10 == 0:
                future_detector = executor.submit(detector.inference_multithread, frame_top, frame_side)

                if(args.mode == "multiview"): # mobilenet
                    future_segmentor = executor.submit(segmentor.inference_async, frame_top, frame_side, frame_counter)
                else: # mstcn
                    buffer1.append(frame_top)
                    buffer2.append(frame_side)
                    future_segmentor = executor.submit(segmentor.inference, frame_top, frame_side, frame_counter)

            if future_detector is not None and future_detector.done():
                print("detection done")
                detector_result = future_detector.result()
                future_detector = None

            if future_segmentor is not None and future_segmentor.done():
                print("segmentor done")
                segmentor_result = future_segmentor.result()
                future_segmentor = None
                if(args.mode == "multiview"):
                    top_seg_results, side_seg_results = segmentor_result[0], segmentor_result[1]
                else:
                    top_seg_results, side_seg_results = segmentor_result, segmentor_result

            current_time=time.time()
            current_frame = frame_counter
            if (current_time - old_time > interval_second):
                total_frame_processed_in_interval = current_frame - interval_start_frame
                fps = total_frame_processed_in_interval / (current_time - old_time)
                interval_start_frame = current_frame
                old_time = current_time
            print(fps)

            ''' The score evaluation module need to merge the results of the two modules and generate the scores '''
            if detector_result is not None and segmentor_result is not None:
                top_det_results, side_det_results = detector_result[0], detector_result[1]
                state, scoring, keyframe = evaluator.inference(
                        top_det_results = top_det_results,
                        side_det_results = side_det_results,
                        action_seg_results = top_seg_results,
                        frame_top = frame_top,
                        frame_side = frame_side,
                        frame_counter = frame_counter)

                display.display_result(
                        frame_top = frame_top,
                        frame_side = frame_side,
                        side_seg_results = side_seg_results,
                        top_seg_results = top_seg_results,
                        top_det_results = top_det_results,
                        side_det_results = side_det_results,
                        scoring = scoring,
                        state = state,
                        keyframe = keyframe,
                        frame_counter = frame_counter,
                        fps = fps)

        if cv2.waitKey(1) in {ord('q'), ord('Q'), 27}: # Esc
            break


def main():
    args = build_argparser().parse_args()

    buffer1 = deque(maxlen=1000)  # Array buffer
    buffer2 = deque(maxlen=1000)
    core = Core()

    ''' Object Detection Variables'''
    detector = Detector(
            core,
            args.device,
            [args.m_topall, args.m_topmove],
            [args.m_sideall, args.m_sidemove])

    '''Video Segmentation Variables'''
    if(args.mode == "multiview"):
        segmentor = Segmentor(core, args.device, args.m_encoder, args.m_encoder, args.m_decoder)
    elif(args.mode == "mstcn"):
        segmentor = SegmentorMstcn(core, args.device, args.m_encoder, args.m_decoder)

    '''Score Evaluation Variables'''
    evaluator = Evaluator()

    '''Display Obj Detection, Action Segmentation and Score Evaluation Result'''
    display = Display()

    """
        Process the video.
    """
    cap_top = cv2.VideoCapture(args.topview)
    if not cap_top.isOpened():
        raise ValueError(f"Can't read an video or frame from {args.topview}")
    cap_side = cv2.VideoCapture(args.sideview)
    if not cap_side.isOpened():
        raise ValueError(f"Can't read an video or frame from {args.sideview}")

    video_loop(
        args, cap_top, cap_side, detector, segmentor, evaluator, display, buffer1, buffer2)

if __name__ == "__main__":
    main()
