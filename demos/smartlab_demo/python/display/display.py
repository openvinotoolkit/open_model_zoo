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

import numpy as np
import pandas as pd
import cv2

class Display(object):
    def _init_(self):
        '''Score Evaluation Variables'''
        self.video_eval_box = None
        self.eval_vars = None
        self.eval_cbs = None
        self.front_scores_df = None
        self.top_scores_df = None

    def initialize(self):
        # blank page to display score
        self.score_board = np.zeros([200,1920,3],dtype=np.uint8)

    def display_result(self,frame_top,frame_front,front_seg_results,top_seg_results,top_det_results,front_det_results,scoring,state,frame_counter):
        #renew score board so that when put cv2.puttext text will not overlap
        self.score_board = np.zeros([200,1920,3],dtype=np.uint8)

        #add action name of each frame at middle top
        cv2.putText(frame_top, front_seg_results, (700, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=1.5, thickness=3)
        cv2.putText(frame_front, top_seg_results, (700, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=1.5, thickness=3)

        #display frame_number at top left corner
        cv2.putText(frame_top, "frame_"+str(frame_counter).zfill(6), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=1.5, thickness=3)
        cv2.putText(frame_front, "frame"+str(frame_counter).zfill(6), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=1.5, thickness=3)

        # show current state for troubleshooting purpose
        cv2.putText(frame_top, state, (1500, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=1.5, thickness=3)
        cv2.putText(frame_front, state, (1500, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=1.5, thickness=3)

        #display obj detection result for both view
        for row, obj_cls in zip(top_det_results[0], top_det_results[1]):
            x_min = int(row[0])
            y_min = int(row[1])
            x_max = int(row[2])
            y_max = int(row[3])
            obj = obj_cls

            cv2.putText(frame_top, obj, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=0.9, thickness=2)
            frame_top = cv2.rectangle(frame_top, (x_min,y_min),(x_max,y_max), color=(255,0,0), thickness=2)

        for row, obj_cls in zip(front_det_results[0], front_det_results[1]):
            x_min = int(row[0])
            y_min = int(row[1])
            x_max = int(row[2])
            y_max = int(row[3])
            obj = obj_cls

            cv2.putText(frame_front, obj, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), fontScale=0.9, thickness=2)
            frame_front = cv2.rectangle(frame_front, (x_min,y_min),(x_max,y_max), color=(255,0,0), thickness=2)

        # display scoring
        i_rider = scoring['initial_score_rider']
        i_balance = scoring['initial_score_balance']
        m_rider_tweezers = scoring['measuring_score_rider_tweezers']
        m_balance = scoring['measuring_score_balance']
        m_object_left = scoring['measuring_score_object_left']
        m_weights_right_tweezers = scoring['measuring_score_weights_right_tweezers']
        m_weights_order = scoring['measuring_score_weights_order']
        e_tidy = scoring['end_score_tidy']

        cv2.putText(self.score_board, f"Initial : rider[{i_rider}] balance[{i_balance}]", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), fontScale=0.9, thickness=2)
        cv2.putText(self.score_board, f"Measuring : object[{m_object_left}] weights_t[{m_weights_right_tweezers}] order[{m_weights_order}] rider_t[{m_rider_tweezers}] balance[{m_balance}]", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), fontScale=0.9, thickness=2)
        cv2.putText(self.score_board, f"End : tidy[{e_tidy}]", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), fontScale=0.9, thickness=2)

        # resize images and display them side by side, then concatenate with a scoring board to display marks
        frame_top = cv2.resize(frame_top,(int(frame_top.shape[1]/2),int(frame_top.shape[0]/2)))
        frame_front = cv2.resize(frame_front,(int(frame_front.shape[1]/2),int(frame_front.shape[0]/2)))

        result_image = np.concatenate((frame_top, frame_front), axis=1)
        result_image = np.concatenate((result_image, self.score_board), axis=0)
        cv2.imshow("Smart Science Lab",result_image)
