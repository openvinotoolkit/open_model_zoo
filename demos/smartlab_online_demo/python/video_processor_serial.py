# -*- coding: utf-8 -*-

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

import os
import cv2
import time
import pandas as pd
import numpy as np
import threading
from collections import deque
from argparse import ArgumentParser, SUPPRESS
from object_detection.detector import Detector
from temporal_segmentation.segmentor import Segmentor
from score_evaluation.evaluator import Evaluator
from display.display import Display


class Application(object):
    """
        Video process for the pre-recorder video (in this version we aim to mimic the online process )
    """
    def __init__(self, ):
        """
            Initialize Variables
        """
        self.playing = True  # Control button for video processing
        self.frame_counter =  0 # Frame index counter
        self.buffer_top = deque(maxlen=1000)  # Array buffer
        self.buffer_front = deque(maxlen=1000)

        ''' Progress Variables'''
        self.det_process_counter = 0  # Number of frames processed by the object detection module
        self.seg_process_counter = 0  # Number of frames processed by the temporal segmentation module
        self.eval_process_counter = 0  # Number of frames processed by the score evaluation module

        ''' Object Detection Variables'''
        self.detector = Detector(
                [args.m_topall, args.m_topmove],
                [args.m_frontall, args.m_frontmove],
                False)
        self.detector.initialize()  # Initialize the session and load the model parameters

        '''Video Segmentation Variables'''
        if(args.mode == "multiview"):
            self.segmentor = Segmentor(args.m_encoder, args.m_decoder)
            self.segmentor.initialize()  # Initialize the session and load the model parameters
        elif(args.mode == "mstcn"):
            self.segmentor = Segmentor(args.m_i3d, args.m_mstcn)
            self.segmentor.initialize()  # Initialize the session and load the model parameters

        '''Score Evaluation Variables'''
        self.evaluator = Evaluator()
        self.evaluator.initialize()

        '''Display Obj Detection, Action Segmentation and Score Evaluation Result'''
        self.display = Display()
        self.display.initialize()

    def video_parser(self, top_video_path, front_video_path):
        """
            Process the video.
        """
        self.cap_top = cv2.VideoCapture(top_video_path)
        self.cap_front = cv2.VideoCapture(front_video_path)

        while self.cap_top.isOpened() and self.cap_front.isOpened() and self.playing:
            ret_top, frame_top = self.cap_top.read()  # frame:480 x 640 x 3
            ret_front, frame_front = self.cap_front.read()

            if ret_top and ret_front:
                self.buffer_top.append(cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB))
                self.buffer_front.append(cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB))
                self.frame_counter += 1

                print(self.frame_counter)
                ''' The object detection module need to generate detection results(for the current frame) '''
                top_det_results, front_det_results = self.detector.inference(
                        img_top=frame_top, img_front=frame_front)

                ''' The temporal segmentation module need to self judge and generate segmentation results for all historical frames '''
                if(args.mode == "multiview"):
                    top_seg_results, front_seg_results = self.segmentor.inference(
                            buffer_top=frame_top,
                            buffer_front=frame_front,
                            frame_index=self.frame_counter
                            )
                elif(args.mode == "mstcn"):
                    top_seg_results, front_seg_results = self.segmentor.inference(
                            buffer_top=frame_top,
                            buffer_front=frame_front,
                            frame_index=self.frame_counter
                            )

                ''' The score evaluation module need to merge the results of the two modules and generate the scores '''
                self.state, self.scoring = self.evaluator.inference(
                        top_det_results=top_det_results,
                        front_det_results=front_det_results,
                        top_seg_results=top_seg_results,
                        front_seg_results=front_seg_results,
                        frame_top=frame_top,
                        frame_front=frame_front
                        )

                self.display.display_result(
                        frame_top=frame_top,
                        frame_front=frame_front,
                        front_seg_results=front_seg_results,
                        top_seg_results=top_seg_results,
                        top_det_results=top_det_results,
                        front_det_results=front_det_results,
                        scoring=self.scoring,
                        state=self.state,
                        frame_counter=self.frame_counter)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):     #press 'q' to exit
                    break
            else:
                print("Finished !")
                break

    def get_video_fps(self, cap):
        return cap.get(cv2.CAP_PROP_FPS)

    def get_video_total_frames(self, cap):
        return cap.get(cv2.CAP_PROP_FRAME_COUNT)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                    help='Show this help message and exit.')
    args.add_argument('-tv', '--topview', required=True,
                    help='Required. Topview stream to be processed. The input must be a single image, '
                    'a folder of images, video file or camera id.')
    args.add_argument('-fv', '--frontview', required=True,
                    help='Required. FrontView to be processed. The input must be a single image, '
                    'a folder of images, video file or camera id.')
    args.add_argument('-m_ta', '--m_topall', help='Required. Path to topview all class model.', required=True, type=str)
    args.add_argument('-m_tm', '--m_topmove', help='Required. Path to topview moving class model.', required=True, type=str)
    args.add_argument('-m_fa', '--m_frontall', help='Required. Path to frontview all class model.', required=True, type=str)
    args.add_argument('-m_fm', '--m_frontmove', help='Required. Path to frontview moving class model.', required=True, type=str)

    subparsers = parser.add_subparsers(help='sub-command help')
    args_mutiview = subparsers.add_parser('multiview', help='multiview help')
    args_mutiview.add_argument('-m_en', '--m_encoder', help='Required. Path to encoder model.', required=True, type=str)
    args_mutiview.add_argument('-m_de', '--m_decoder', help='Required. Path to decoder model.', required=True, type=str)
    args_mutiview.add_argument('--mode', default='multiview', help='Option. Path to decoder model.', type=str)
    args_mstcn = subparsers.add_parser('mstcn', help='mstcm help')
    args_mstcn.add_argument('-m_i3d', '--m_i3d', help='Required. Path to i3d model.', required=True, type=str)
    args_mstcn.add_argument('-m_mstcn', '--m_mstcn', help='Required. Path to mstcn model.', required=True, type=str)
    args_mstcn.add_argument('--mode', default='mstcn', help='Option. Path to decoder model.', type=str)

    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()

    application = Application()
    application.video_parser(top_video_path=args.topview,
                             front_video_path=args.frontview)