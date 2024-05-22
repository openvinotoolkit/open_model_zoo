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

import numpy as np
import cv2
from pathlib import Path


class Display:
    def __init__(self, show):
        '''Score Evaluation Variables'''
        self.show = show
        self.wait_icon = cv2.imread(str(Path(__file__).resolve().parents[0] / 'icon/wait.png'), cv2.IMREAD_UNCHANGED)
        self.no_icon = cv2.imread(str(Path(__file__).resolve().parents[0] / 'icon/no.png'), cv2.IMREAD_UNCHANGED)
        self.done_icon = cv2.imread(str(Path(__file__).resolve().parents[0] / 'icon/done.png'), cv2.IMREAD_UNCHANGED)
        self.colour_map = {
            "noise_action": [127, 127, 127],
            "put_take": [0, 0, 255],
            "adjust_rider": [255, 0, 0],
            "remove_support_sleeve": [47, 79, 79],
            "open_box": [72, 61, 139],
            "adjust_nut": [46, 139, 87],
            "adjust_balancing": [255, 20, 147],
            "put_left": [160, 32, 240],
            "choose_weight": [139, 119, 101],
            "put_right": [139, 131, 134],
            "take_right": [139, 58, 98],
            "close_box": [176, 226, 255],
            "take_left": [139, 0, 0],
            "install_support_sleeve": [238, 0, 0],
            None: [0, 0, 0]}
        self.screen_width = 1920
        self.screen_height = 1080
        self.screen_width_half = 960
        self.screen_height_half = 540
        self.barHeight = 50
        self.segmentationBar = np.zeros((self.barHeight, self.screen_width, 3))
        self.segmentationBar[20:23, ::10] = 255
        self.segmentationBar[:, -1] = 255
        self.w1 = self.screen_width // 16
        self.w2 = self.screen_width_half + self.w1

        # renew score board so that when put cv2.puttext text will not overlap
        self.score_board = np.zeros([self.screen_height_half, self.screen_width, 3], dtype=np.uint8)

    def draw_text(self, img, text,
                  font=cv2.FONT_HERSHEY_TRIPLEX,
                  pos=(0, 0),
                  font_scale=1,
                  font_thickness=1,
                  text_color=(255, 255, 255),
                  text_color_bg=(0, 0, 0),
                  icon=None
                  ):
        distance_text_and_icon = 30
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, (x - 10 - distance_text_and_icon, y - 10),
                      (x + 700, y + text_h + 10), text_color_bg, -1)
        self.score_board[y: y + 22, x - distance_text_and_icon: x + 22 - distance_text_and_icon] = cv2.resize(
            icon[:, :, :3], (text_h, text_h), interpolation=cv2.INTER_AREA)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font,
                    font_scale, text_color, font_thickness)
        return text_size

    def draw_text_without_background(self, img, text,
                                     font=cv2.FONT_HERSHEY_TRIPLEX,
                                     pos=(0, 0),
                                     font_scale=1,
                                     font_thickness=2,
                                     text_color=(255, 255, 255),
                                     text_color_bg=(0, 0, 0)
                                     ):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
        return text_size

    def display_result(self, frame_top, frame_side, side_seg_results, top_seg_results,
                       top_det_results, side_det_results, scoring, state, keyframe, frame_counter, fps):

        if state == 'Initial':
            self.score_board = np.zeros([self.screen_height_half, self.screen_width, 3], dtype=np.uint8)
            display_status = 'Setting Up ...'
            initial_text_color_bg = (255, 0, 0)
            measuring_text_color_bg = (128, 128, 128)
            initial_icon = self.wait_icon
            measuring_icon = self.no_icon
        elif state == 'Measuring':
            self.score_board = np.zeros([self.screen_height_half, self.screen_width, 3], dtype=np.uint8)
            display_status = 'Set Up Done. Evaluating Measuring Phase...'
            initial_text_color_bg = (0, 180, 0)
            measuring_text_color_bg = (self.screen_height_half, 0, 0)
            initial_icon = self.done_icon
            measuring_icon = self.wait_icon
        elif state == 'Finish':
            self.score_board = np.zeros([self.screen_height_half, self.screen_width, 3], dtype=np.uint8)
            count = 0
            for item in scoring.values():
                if item == 1:
                    count += 1
            total_score_obtained = count
            display_status = f'Experiment completed. Total Score is {total_score_obtained}/8'
            initial_text_color_bg = (0, 180, 0)
            measuring_text_color_bg = (0, 180, 0)
            initial_icon = self.done_icon
            measuring_icon = self.done_icon

        # Add action name of each frame at middle top
        cv2.putText(frame_top, side_seg_results, (700, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, color=self.colour_map[side_seg_results],
                    fontScale=1.5, thickness=3)
        cv2.putText(frame_side, top_seg_results, (700, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, color=self.colour_map[top_seg_results],
                    fontScale=1.5, thickness=3)

        # display FPS at top left corner
        cv2.putText(frame_top, f"FPS: {fps: .2f}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                    fontScale=1.5, thickness=3)
        cv2.putText(frame_side, f"FPS: {fps: .2f}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                    fontScale=1.5, thickness=3)

        # display obj detection result for both view
        if top_det_results[0] is not None:
            for row, obj_cls in zip(top_det_results[0], top_det_results[2]):
                x_min = int(row[0])
                y_min = int(row[1])
                x_max = int(row[2])
                y_max = int(row[3])

                cv2.putText(frame_top, obj_cls, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                            fontScale=0.9, thickness=2)
                frame_top = cv2.rectangle(frame_top, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)

        if side_det_results[0] is not None:
            for row, obj_cls in zip(side_det_results[0], side_det_results[2]):
                x_min = int(row[0])
                y_min = int(row[1])
                x_max = int(row[2])
                y_max = int(row[3])

                cv2.putText(frame_side, obj_cls, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                            fontScale=0.9, thickness=2)
                frame_side = cv2.rectangle(frame_side, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)

        # display scoring
        i_rider = scoring['initial_score_rider']
        i_balance = scoring['initial_score_balance']
        m_rider_tweezers = scoring['measuring_score_rider_tweezers']
        m_balance = scoring['measuring_score_balance']
        m_object_left = scoring['measuring_score_object_left']
        m_weights_right = scoring['measuring_score_weights_right']
        m_weights_tweezers = scoring['measuring_score_weights_tweezers']
        e_tidy = scoring['end_score_tidy']

        w, h = self.draw_text_without_background(self.score_board, f"SCORE - {display_status}",
                                                 pos=(self.w1, 60), text_color_bg=initial_text_color_bg)
        w, h = self.draw_text(self.score_board, f"INITIALISE RIDER[{i_rider}]",
                              pos=(self.w1, 120), text_color_bg=initial_text_color_bg, icon=initial_icon)
        w, h = self.draw_text(self.score_board, f"INITIALISE BALANCE[{i_balance}]",
                              pos=(self.w2, 120), text_color_bg=initial_text_color_bg, icon=initial_icon)
        w, h = self.draw_text(self.score_board, f"PUT OBJECT ON LEFT TRAY[{m_object_left}]",
                              pos=(self.w1, 180), text_color_bg=measuring_text_color_bg, icon=measuring_icon)
        w, h = self.draw_text(self.score_board, f"PUT WEIGHTS ON RIGHT TRAY[{m_weights_right}]",
                              pos=(self.w2, 180), text_color_bg=measuring_text_color_bg, icon=measuring_icon)
        w, h = self.draw_text(self.score_board, f"MOVE WEIGHTS WITH TWEEZER[{m_weights_tweezers}]",
                              pos=(self.w1, 240), text_color_bg=measuring_text_color_bg, icon=measuring_icon)
        w, h = self.draw_text(self.score_board, f"ADJUST RIDER WITH TWEEZER[{m_rider_tweezers}]",
                              pos=(self.w2, 240), text_color_bg=measuring_text_color_bg, icon=measuring_icon)
        w, h = self.draw_text(self.score_board, f"SCALE IS BALANCED[{m_balance}]",
                              pos=(self.w1, 300), text_color_bg=measuring_text_color_bg, icon=measuring_icon)
        w, h = self.draw_text(self.score_board, f"PUT EQUIPMENTS BACK[{e_tidy}]",
                              pos=(self.w2, 300), text_color_bg=measuring_text_color_bg, icon=measuring_icon)

        # draw action segmentation bar
        self.segmentationBar[:, :-1] = self.segmentationBar[:, 1:]
        self.segmentationBar[:, -1] = np.asarray(self.colour_map[top_seg_results])
        if frame_counter % 10 == 0:  # add keyframe
            self.segmentationBar[20: 23, -1, :] = 255
        self.score_board[:self.barHeight, :] = self.segmentationBar[:, :]

        # resize images and display them side by side, then concatenate with a scoring board to display marks
        frame_top = cv2.resize(
            frame_top, (self.screen_width_half, self.screen_height_half))
        frame_side = cv2.resize(
            frame_side, (self.screen_width_half, self.screen_height_half))

        result_image = np.concatenate((frame_top, frame_side), axis=1)
        result_image = np.concatenate((result_image, self.score_board), axis=0)

        if self.show:
            cv2.imshow("Smart Science Lab", result_image)
