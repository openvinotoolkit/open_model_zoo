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
import cv2


class Display:
    def __init__(self):
        '''Score Evaluation Variables'''

    def display_result(self, frame_top, frame_side, side_seg_results, top_seg_results, \
        top_det_results, side_det_results, scoring, state, keyframe, frame_counter, fps):

        #renew score board so that when put cv2.puttext text will not overlap
        self.score_board = np.zeros([200, 1920, 3], dtype=np.uint8)

        #add action name of each frame at middle top
        cv2.putText(frame_top, side_seg_results, (700, 80), cv2.FONT_HERSHEY_SIMPLEX, color = (0, 0, 255), fontScale = 1.5, thickness = 3)
        cv2.putText(frame_side, top_seg_results, (700, 80), cv2.FONT_HERSHEY_SIMPLEX, color = (0, 0, 255), fontScale = 1.5, thickness = 3)

        #display frame_number at top left corner
        cv2.putText(frame_top, f"frame{frame_counter: 6d}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale = 1.5, thickness = 3)
        cv2.putText(frame_side, f"frame{frame_counter: 6d}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale = 1.5, thickness = 3)

        #display FPS at top left corner
        cv2.putText(frame_top, f"FPS: {fps: .2f}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale = 1.5, thickness = 3)
        cv2.putText(frame_side, f"FPS: {fps: .2f}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale = 1.5, thickness = 3)

        # show current state for troubleshooting purpose
        cv2.putText(frame_top, state, (1500, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale = 1.5, thickness = 3)
        cv2.putText(frame_side, state, (1500, 80), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale = 1.5, thickness = 3)

        #display obj detection result for both view
        for row, obj_cls in zip(top_det_results[0], top_det_results[2]):
            x_min = int(row[0])
            y_min = int(row[1])
            x_max = int(row[2])
            y_max = int(row[3])

            cv2.putText(frame_top, obj_cls, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale = 0.9, thickness = 2)
            frame_top = cv2.rectangle(frame_top, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)

        for row, obj_cls in zip(side_det_results[0], side_det_results[2]):
            x_min = int(row[0])
            y_min = int(row[1])
            x_max = int(row[2])
            y_max = int(row[3])

            cv2.putText(frame_side, obj_cls, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, color = (0, 0, 255), fontScale = 0.9, thickness = 2)
            frame_side = cv2.rectangle(frame_side, (x_min, y_min), (x_max, y_max), color = (255, 0, 0), thickness = 2)

        # display scoring
        i_rider = scoring['initial_score_rider']
        i_balance = scoring['initial_score_balance']
        m_rider_tweezers = scoring['measuring_score_rider_tweezers']
        m_balance = scoring['measuring_score_balance']
        m_object_left = scoring['measuring_score_object_left']
        m_weights_right = scoring['measuring_score_weights_right']
        m_weights_tweezers = scoring['measuring_score_weights_tweezers']
        m_weights_order = scoring['measuring_score_weights_order']
        e_tidy = scoring['end_score_tidy']

        # display keyframe
        i_rider_k = keyframe['initial_score_rider']
        i_balance_k = keyframe['initial_score_balance']
        m_rider_tweezers_k = keyframe['measuring_score_rider_tweezers']
        m_balance_k = keyframe['measuring_score_balance']
        m_object_left_k = keyframe['measuring_score_object_left']
        m_weights_right_k = keyframe['measuring_score_weights_right']
        m_weights_tweezers_k = keyframe['measuring_score_weights_tweezers']
        m_weights_order_k = keyframe['measuring_score_weights_order']
        e_tidy_k = keyframe['end_score_tidy']

        cv2.putText(self.score_board, f"Score", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
            color=(255, 255, 255), fontScale = 0.9, thickness = 2)
        cv2.putText(self.score_board, f"Initial : rider[{i_rider}] balance[{i_balance}]",
            (30, 70), cv2.FONT_HERSHEY_SIMPLEX, color = (255, 255, 255), fontScale = 0.9, thickness = 2)
        cv2.putText(self.score_board, f"Measuring : object[{m_object_left}] weights_right[{m_weights_right}] weights_t[{m_weights_tweezers}]",
            (30, 120), cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale = 0.9, thickness = 2)
        cv2.putText(self.score_board, f"            order[{m_weights_order}] rider_t[{m_rider_tweezers}] balance[{m_balance}] tidy[{e_tidy}]",
            (30, 170), cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale = 0.9, thickness = 2)

        cv2.putText(self.score_board, f"Keyframe", (1030, 30), cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), fontScale=0.9, thickness=2)
        cv2.putText(self.score_board, f"Initial : rider[{i_rider_k}] balance[{i_balance_k}]", (1030, 70), cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), fontScale=0.9, thickness=2)
        cv2.putText(self.score_board, f"Measuring : object[{m_object_left_k}] weights_right[{m_weights_right_k}] weights_t[{m_weights_tweezers_k}] ", (1030, 120), cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), fontScale=0.9, thickness=2)
        cv2.putText(self.score_board, f"            order[{m_weights_order_k}] rider_t[{m_rider_tweezers_k}] balance[{m_balance_k}] tidy[{e_tidy_k}]", (1030, 170), cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), fontScale=0.9, thickness=2)

        # resize images and display them side by side, then concatenate with a scoring board to display marks
        frame_top = cv2.resize(
            frame_top, (int(frame_top.shape[1]/2), int(frame_top.shape[0]/2)))
        frame_side = cv2.resize(
            frame_side, (int(frame_side.shape[1]/2), int(frame_side.shape[0]/2)))
        result_image = np.concatenate((frame_top, frame_side), axis=1)
        result_image = np.concatenate((result_image, self.score_board), axis=0)

        cv2.imshow("Smart Science Lab", result_image)