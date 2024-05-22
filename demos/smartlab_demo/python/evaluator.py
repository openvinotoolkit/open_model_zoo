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
import math
import numpy as np


class Evaluator(object):
    def __init__(self):
        """Score Evaluation Variables"""
        # user-defined parameter
        # MS stands for measuring score; IS stands for initial score
        self.multiview_buffer_size = 40  # preprocessing, to filter action
        self.mstcn_batchsize = 24  # preprocessing, to filter action
        self.mstcn_buffer_size = self.mstcn_batchsize * 2
        self.filter_action_threshold = 15  # preprocessing, to filter action (action persist less than this duration will be ignored)
        self.balance_threshold = 12  # [IS_balance & MS_balance]
        self.balance_persist_duration_threshold = 40  # [MS_balance]
        self.rider_portion = 6  # [IS_rider & MS_rider] divide the distance between 2 roundscrew1 into rider_portion portion, if rider falls in the first portion mean rider at zeroth position
        self.rider_move_threshold = 20  # [MS_rider_tweezers] if rider moves more than this value, check if tweezers or hand is used to move rider
        self.buffer_rider_size_limit = 30  # [MS_rider_tweezers]
        self.use_tweezers_threshold = 99  # [MS_rider_tweezers & MS_weights_tweezers] if tweezer and rider/weight distance more than tweezer treshold, consider use hand instead of use tweezer
        self.tweezers_warning_duration = 60  # [MS_rider_tweezers & MS_weights_tweezers] if score related to tweezers is 0 more than this duration/frames, score is 0 and unrevertible; else still revertible
        self.battery_aspect_ratio = 1.9
        self.reset()

    def reset(self):
        self.buffer_pointer = 0
        self.frame_top_buffer = []
        self.frame_side_buffer = []
        self.action_buffer = []
        self.action_mstcn_buffer = []
        self.last_action = None
        self.top_obj_buffer = []
        self.side_obj_buffer = []
        self.frame_counter = 0
        self.top_object_dict = {}
        self.side_object_dict = {}
        self.is_put_take_observed = False
        self.state = "Initial"
        self.mode = 'multiview'
        self.buffer_size = self.multiview_buffer_size
        self.buffer_rider = []  # buffer store coordinate of rider and tweezers to detect the move of rider, to evaluate the use of tweezers when adjust rider
        self.tweezers_warning = None
        self.is_object_put = False  # if battery is put on tray no matter left or right, return True
        self.object_direction = None  # if initially object put in left, then change to right or vice versa, use this parameter to know the change to give mark again
        self.is_weights_put = False  # if weight is put on tray no matter left or right, return True
        self.weights_direction = None  # if initially weights put in right, then change to left or vice versa, use this parameter to know the change to give mark again
        self.can_tidy = False  # if battery and weight has been put on the tray(no matter left/right), then can tidy becomes true
        self.rider_zero = False
        self.num_weight_inside_tray = 0  # use in weight order evaluation -- only evaluate order when there is changes in this parameter
        self.record_weight_inside_tray = []  # use in weight order evaluation -- only evaluate order when there is changes in this parameter
        self.rider_tweezers_lock_mark = False
        self.weight_order_lock_mark = False
        self.is_weight_tweezers_lock_mark = False
        self.is_change_object_direction = False  # mark given for object left will not change once given, except when this parameter become true
        self.is_change_weights_direction = False
        self.balance_persist_duration = 0
        self.measuring_state_balance_lock_mark = False  # once this variable is True, even the balance is not balanced, measuring state balance mark still is given. \
        # it only turns true if balance persists more than self.balance_persist_duration_threshold frames
        # however, when weights are added to the tray, this lock will become False again

        self.scoring = {
            "initial_score_rider": 0,
            "initial_score_balance": 0,
            "measuring_score_rider_tweezers": 0,
            "measuring_score_balance": 0,
            "measuring_score_object_left": 0,
            "measuring_score_weights_right": 0,
            "measuring_score_weights_tweezers": 0,
            "measuring_score_weights_order": 0,
            "end_score_tidy": 0
        }

        # keyframe
        self.keyframe = {
            "initial_score_rider": 0,
            "initial_score_balance": 0,
            "measuring_score_rider_tweezers": 0,
            "measuring_score_balance": 0,
            "measuring_score_object_left": 0,
            "measuring_score_weights_right": 0,
            "measuring_score_weights_tweezers": 0,
            "measuring_score_weights_order": 0,
            "end_score_tidy": 0
        }

    def inference(self,
                  top_det_results,
                  side_det_results,
                  action_seg_results,
                  frame_top,
                  frame_side,
                  frame_counter,
                  mode):
        """
        Args:
            top_det_results:
            side_det_results:
            action_seg_results:
            action_seg_results:
        Returns:
            Progress of the frame index
        """
        self.frame_counter = frame_counter
        self.mode = mode
        if self.mode == 'multiview':
            self.buffer_size = self.multiview_buffer_size
        else:  # mstcn
            self.buffer_size = self.mstcn_buffer_size
        action_seg_results = self.filter_action(action_seg_results, mode=mode)

        top_det_results, side_det_results = self.filter_object(top_det_results, side_det_results)
        self.filter_image(frame_top, frame_side)

        if frame_counter > self.buffer_size:
            self.classify_state(action_seg_results)
            self.top_object_dict = self.get_object(det_results=top_det_results)
            self.side_object_dict = self.get_object(det_results=side_det_results)

            if self.state == "Initial":
                self.evaluate_rider()
                self.evaluate_scale_balance()

            elif self.state == "Measuring":
                # self.evaluate_object_left_from_view(view='top')
                if action_seg_results in ["put_take", "put_left", "put_right", "take_left", "take_right"]:
                    self.evaluate_object_left()
                    self.evaluate_weights_right()
                    if not self.can_tidy:
                        self.evaluate_end_state()
                self.evaluate_scale_balance()

                if action_seg_results == "adjust_rider":
                    self.evaluate_rider_tweezers()

                if self.is_object_put and self.is_weights_put:  # if object and weights are put no matter at left/right tray,
                    self.evaluate_end_tidy()  # then can start evaluate whether they keep the apparatus

            self.check_score_validity()

        else:
            action_seg_results = None

        return self.state, self.scoring, self.keyframe, action_seg_results

    def check_consecutive(self):
        """
        Given a list: ['noise','noise','noise','put_take','noise','noise','noise'], this function calculate number of consecutive element, store in count_list
        and return the first action that persist more than filter_action_threshold frame

        count_list = [('noise', 3), ('put_take', 1), ('adjust_rider', 3)]
        return 'noise' (as is the first action persist more than filter_action_threshold frame, assume threshold is 3)
        """
        if not self.action_buffer:
            return 'noise_action'

        count_list = []
        count = 1
        for i in range(len(self.action_buffer) - 1):
            if self.action_buffer[i] == self.action_buffer[i + 1]:
                count += 1
            elif self.action_buffer[i] is not self.action_buffer[i + 1]:
                count_list.append((self.action_buffer[i], count))
                count = 1
            if i == len(self.action_buffer) - 2:
                count_list.append((self.action_buffer[i], count))

        for i in range(len(count_list)):
            if count_list[i][1] >= self.filter_action_threshold:
                return count_list[i][0]
            if i == len(count_list) - 1:
                return 'noise_action'

    def filter_action(self, action_seg_results, mode):
        """filter action segmentation result. action which persists less than filter_action_threshold frame will be ignored"""
        if mode == 'multiview':
            # at beginning of evaluation (less than self.buffer_size frame), action is None
            if len(self.action_buffer) < self.multiview_buffer_size:
                self.action_buffer.append(action_seg_results)
                return None
            # after self.buffer_size frame, filtering of action occurs
            elif len(self.action_buffer) == self.multiview_buffer_size:
                if self.action_buffer[0] is not self.last_action:
                    self.last_action = self.check_consecutive()
                self.action_buffer.pop(0)
                self.action_buffer.append(action_seg_results)
                return self.last_action
        elif mode == 'mstcn':
            if action_seg_results is not None:
                self.action_mstcn_buffer.extend(action_seg_results)
            if len(self.action_mstcn_buffer) > self.mstcn_buffer_size:  # 48
                self.action_mstcn_buffer = self.action_mstcn_buffer[self.mstcn_batchsize:]  # self.mstcn_batchsize: 24
                self.buffer_pointer = 0

            if len(self.action_mstcn_buffer) == self.mstcn_buffer_size:
                self.buffer_pointer += 1
                self.action_buffer = self.action_mstcn_buffer[
                                     self.buffer_pointer: self.buffer_pointer + self.mstcn_batchsize]
                if self.action_buffer[0] is not self.last_action:
                    self.last_action = self.check_consecutive()
                return self.last_action
            else:
                return None

    def filter_object(self, top_det_results, side_det_results):
        first_top_det_results = first_side_det_results = None
        # --- corner case 1, top_det_results: battery is detected as weight. Use aspect ratio to remove wrong battery
        battery_index = [i for i in range(len(top_det_results[2])) if top_det_results[2][i] == 'battery']
        if len(battery_index) > 1:
            removed_index = []
            for i in battery_index:
                width = top_det_results[0][i][0] - top_det_results[0][i][2]
                height = top_det_results[0][i][1] - top_det_results[0][i][3]
                ratio = max(width / height, height / width)
                # should remove wrong battery
                if ratio < self.battery_aspect_ratio:
                    removed_index.append(i)
            for i in range(len(removed_index) - 1, -1, -1):
                for index, item in enumerate(top_det_results):
                    if index == 2:
                        item.pop(removed_index[i])
                    else:
                        np.delete(item, removed_index[i])
        # --- corner case 1, top_det_results: battery is detected as weight. Use aspect ratio to remove wrong battery
        if self.mode == 'multiview':
            if len(self.top_obj_buffer) < self.buffer_size:
                self.top_obj_buffer.append(top_det_results)
                self.side_obj_buffer.append(side_det_results)
                first_top_det_results = first_side_det_results = [np.array([[100, 500, 100, 500]]), [],
                                                                  np.array(['balance']), ]
            elif len(self.top_obj_buffer) == self.buffer_size:
                first_top_det_results = self.top_obj_buffer.pop(0)
                first_side_det_results = self.side_obj_buffer.pop(0)
                self.top_obj_buffer.append(top_det_results)
                self.side_obj_buffer.append(side_det_results)
            else:
                print('len(self.top_obj_buffer) > self.buffer_size')
            return first_top_det_results, first_side_det_results

        else:  # mstcn
            if len(self.top_obj_buffer) < self.mstcn_buffer_size:
                self.top_obj_buffer.append(top_det_results)
                self.side_obj_buffer.append(side_det_results)
                first_top_det_results = first_side_det_results = [np.array([[100, 500, 100, 500]]), [],
                                                                  np.array(['balance']), ]
            elif len(self.top_obj_buffer) == self.mstcn_buffer_size:
                first_top_det_results = self.top_obj_buffer.pop(0)
                first_side_det_results = self.side_obj_buffer.pop(0)
                self.top_obj_buffer.append(top_det_results)
                self.side_obj_buffer.append(side_det_results)
            else:
                print('algorithm error: len(self.top_obj_buffer) > self.mstcn_buffer_size')
            return first_top_det_results, first_side_det_results

    def filter_image(self, frame_top, frame_side):
        if self.mode == 'multiview':
            if len(self.frame_top_buffer) < self.buffer_size:
                self.frame_top_buffer.append(frame_top)
                self.frame_side_buffer.append(frame_side)
                h, w, c = frame_top.shape
                blank_image = 255 * np.ones(shape=(h, w, c), dtype=np.uint8)
                first_frame_top = first_frame_side = blank_image
            elif len(self.frame_top_buffer) == self.buffer_size:
                first_frame_top = self.frame_top_buffer.pop(0)
                first_frame_side = self.frame_side_buffer.pop(0)
                self.frame_top_buffer.append(frame_top)
                self.frame_side_buffer.append(frame_side)
            else:
                first_frame_top = first_frame_side = None
                print('algorithm error: len(self.frame_top_buffer) > self.buffer_size')

            return first_frame_top, first_frame_side

        elif self.mode == 'mstcn':
            if len(self.frame_top_buffer) < self.mstcn_buffer_size:
                self.frame_top_buffer.append(frame_top)
                self.frame_side_buffer.append(frame_side)
                h, w, c = frame_top.shape
                blank_image = 255 * np.ones(shape=(h, w, c), dtype=np.uint8)
                first_frame_top = first_frame_side = blank_image
            elif len(self.frame_top_buffer) == self.mstcn_buffer_size:
                first_frame_top = self.frame_top_buffer.pop(0)
                first_frame_side = self.frame_side_buffer.pop(0)
                self.frame_top_buffer.append(frame_top)
                self.frame_side_buffer.append(frame_side)
            else:
                first_frame_top = first_frame_side = None
            return first_frame_top, first_frame_side

    def get_object(self, det_results):

        """
        Most object return list of obj coordinate in dict except weight
        most object:
        return a dictionary of list of object coordinate,
        eg.[array([x_min,y_min,x_max,y_max]),...]

        weight:
            weights_obj_coor return list of object name and coordinate,
            eg.[[[weights5_g,array([x_min,y_min,x_max,y_max])],...]
            as weight (5g,10g) etc is important for weight_order algo

        Parameters
        ----------
        det_results: list
            det_results[0]: [[x_min, y_min, x_max, y_max], ... [...]]
            det_results[1]: obj label ids
            det_results[2]: obj labels
            det_results[3]: obj scores

        Returns
        -------
        m_object: tuple of dict
        """

        rider_coor = []
        balance_coor = []
        pointerhead_coor = []
        pointer_coor = []
        roundscrew1_coor = []
        roundscrew2_coor = []
        tray_coor = []
        pointer_sleeve_coor = []
        support_sleeve_coor = []
        ruler_coor = []
        scale_coor = []
        box_coor = []
        battery_coor = []
        tweezers_coor = []
        weights_obj_coor = []

        if det_results[0] is not None:
            for obj, coor in zip(det_results[2], det_results[0]):
                if obj == 'rider':
                    rider_coor.append(coor)
                elif obj == 'balance':
                    balance_coor.append(coor)
                elif obj == 'pointerhead':
                    pointerhead_coor.append(coor)
                elif obj == 'pointer':
                    pointer_coor.append(coor)
                elif obj == 'roundscrew1':
                    roundscrew1_coor.append(coor)
                elif obj == 'roundscrew2':
                    roundscrew2_coor.append(coor)
                elif obj == 'tray':
                    tray_coor.append(coor)
                elif obj == 'pointer_sleeve':
                    pointer_sleeve_coor.append(coor)
                elif obj == 'support_sleeve':
                    support_sleeve_coor.append(coor)
                elif obj == 'ruler':
                    ruler_coor.append(coor)
                elif obj == 'scale':
                    scale_coor.append(coor)
                elif obj == 'battery':
                    battery_coor.append(coor)
                elif obj == 'balance':
                    balance_coor.append(coor)
                elif obj == 'box':
                    box_coor.append(coor)
                elif obj == 'tweezers':
                    tweezers_coor.append(coor)
                elif obj == 'tray':
                    tray_coor.append(coor)
                elif obj == 'weight_5g':
                    weights_obj_coor.append([obj, coor])
                elif obj == 'weight_10g':
                    weights_obj_coor.append([obj, coor])
                elif obj == 'weight_20g':
                    weights_obj_coor.append([obj, coor])
                elif obj == 'weight_50g':
                    weights_obj_coor.append([obj, coor])
                elif obj == 'weight_100g':
                    weights_obj_coor.append([obj, coor])
                elif obj == 'weights':
                    weights_obj_coor.append([obj, coor])

        if self.state == 'Initial':
            i_object = {'rider': rider_coor,
                        'pointer': pointer_coor,
                        'pointerhead': pointerhead_coor,
                        'roundscrew1': roundscrew1_coor,
                        'roundscrew2': roundscrew2_coor,
                        'support_sleeve': support_sleeve_coor,
                        'scale': scale_coor,
                        'pointer': pointer_coor}
            return (i_object)
        elif self.state == 'Measuring':
            m_object = {'rider': rider_coor,
                        'pointer': pointer_coor,
                        'pointerhead': pointerhead_coor,
                        'roundscrew1': roundscrew1_coor,
                        'roundscrew2': roundscrew2_coor,
                        'battery': battery_coor,
                        'balance': balance_coor,
                        'support_sleeve': support_sleeve_coor,
                        'pointer_sleeve': pointer_sleeve_coor,
                        'tray': tray_coor,
                        'tweezers': tweezers_coor,
                        'weights': weights_obj_coor,
                        'scale': scale_coor,
                        'pointer': pointer_coor}
            return (m_object)

    def get_center_coordinate(self, coor):
        [x_min, y_min, x_max, y_max] = coor
        center_coor = (x_min + x_max) / 2, (y_min + y_max) / 2
        return center_coor

    def is_inside(self, small_item_center_coor, big_item_coor):
        [big_x_min, big_y_min, big_x_max, big_y_max] = big_item_coor
        [small_center_x, small_center_y] = small_item_center_coor
        if big_x_min <= small_center_x <= big_x_max and big_y_min < small_center_y < big_y_max:
            return True
        else:
            return False

    def is_behind(self, small_item_coor, big_item_coor):
        [big_x_min, big_y_min, big_x_max, big_y_max] = big_item_coor
        [small_x_min, small_y_min, small_x_max, small_y_max] = small_item_coor
        if (big_y_min + big_y_max) / 2 < small_y_min:
            return True
        return False

    def classify_state(self, action_seg_results):
        """
        filter input data for action so that only action persists
        more than certain frame taken as true data

        Parameters
        ----------
        action_seg_results: str
            action label
        """
        if action_seg_results in ["put_left", "put_right", "take_left", "take_right", "put_take"]:
            self.is_put_take_observed = True
            self.state = "Measuring"
        elif not self.is_put_take_observed:
            self.state = "Initial"

    def rotate(self, left, right, center):
        """
        Given 3 points, fix the left point, rotate until the right point is horizontal to the left point
        Return the rotated coordinate of the three points

        Used to correct angle in adjust_rider (roundscrew1 and rider) and balance(roundscrew2 and pointerhead)
        """
        [left_x, left_y] = left
        [right_x, right_y] = right
        [center_x, center_y] = center

        # theta is the angle need to be rotated so that two roundscrew2 is horizontal
        if right_x - left_x != 0:  # to avoid num/zero = infinity
            if right_y <= left_y:
                theta = abs(math.atan((right_y - left_y) / (right_x - left_x)))  # angle in rad by default
            elif right_y > left_y:
                theta = -abs(math.atan((right_y - left_y) / (right_x - left_x)))
        else:
            theta = 0
        # offset to make left roundscrew coordinate as (0,0) so that we can use rotation matrix
        offset_x = left_x
        offset_y = left_y

        rotated_left_coor = int(
            ((left_x - offset_x) * math.cos(theta) - (left_y - offset_y) * math.sin(theta)) + offset_x), \
                            int(((left_x - offset_x) * math.sin(theta) + (left_y - offset_y) * math.cos(
                                theta)) + offset_y)
        rotated_right_coor = int(
            ((right_x - offset_x) * math.cos(theta) - (right_y - offset_y) * math.sin(theta)) + offset_x), \
                             int(((right_x - offset_x) * math.sin(theta) + (right_y - offset_y) * math.cos(
                                 theta)) + offset_y)
        rotated_center_coor = int(
            ((center_x - offset_x) * math.cos(theta) - (center_y - offset_y) * math.sin(theta)) + offset_x), \
                              int(((center_x - offset_x) * math.sin(theta) + (center_y - offset_y) * math.cos(
                                  theta)) + offset_y)

        return rotated_left_coor, rotated_right_coor, rotated_center_coor

    def evaluate_rider(self):
        """
        Function:
            To evaluate whether rider is pushed to zero position
        """
        roundscrew1_coor = self.side_object_dict['roundscrew1']
        rider_coor = self.side_object_dict['rider']
        # only evaluate rider_zero if 2 roundscrew1 and 1 rider are found
        if len(roundscrew1_coor) == 2 and len(rider_coor) == 1:
            # find center coordinate of rider and roundscrew1
            (x0, y0) = \
                ((roundscrew1_coor[0][2] + roundscrew1_coor[0][0]) / 2,
                 (roundscrew1_coor[0][3] + roundscrew1_coor[0][1]) / 2)
            (x1, y1) = \
                ((roundscrew1_coor[1][2] + roundscrew1_coor[1][0]) / 2,
                 (roundscrew1_coor[1][3] + roundscrew1_coor[1][1]) / 2)
            rider_center_coor = \
                ((rider_coor[0][2] + rider_coor[0][0]) / 2, (rider_coor[0][3] + rider_coor[0][1]) / 2)

            # determine left/right roundscrew
            if x0 < x1:
                left_roundscrew1_center_coor = [x0, y0]
                right_roundscrew1_center_coor = [x1, y1]
            else:
                left_roundscrew1_center_coor = [x1, y1]
                right_roundscrew1_center_coor = [x0, y0]

            # rotate to make two roundscrew1 in a horizontal line
            rotated_left_coor, rotated_right_coor, rotated_center_coor = \
                self.rotate(left=left_roundscrew1_center_coor, right=right_roundscrew1_center_coor,
                            center=rider_center_coor)
            limit = rotated_left_coor[0] + (rotated_right_coor[0] - rotated_left_coor[0]) / self.rider_portion

            # if rider center position < 1/10 of length between 2 roundscrew, consider rider is pushed to zero position
            if rotated_center_coor[0] < limit:
                if self.state == "Initial":
                    self.scoring["initial_score_rider"] = 1
                    self.keyframe["initial_score_rider"] = self.frame_counter
                elif self.state == 'Measuring':
                    self.rider_zero = True  # self.rider_zero is to determine end_state score
            else:
                if self.state == "Initial":
                    self.scoring["initial_score_rider"] = 0
                    self.keyframe["initial_score_rider"] = self.frame_counter
                elif self.state == 'Measuring':
                    self.rider_zero = False

    def evaluate_rider_tweezers(self):
        """
        Function:
            To evaluate whether rider is pushed using tweezers

        Logic:
            if rider moves, tweezers coordinate should within certain pixels (defined in self.use_tweezers_threshold) from the rider coordinate

        """
        rider_coor = self.side_object_dict['rider']
        tweezers_coor = self.side_object_dict['tweezers']

        # only evaluate rider_tweezers if 1 rider and 1 tweezers are found
        if len(rider_coor) == 1:
            rider_min_coordinate = np.array(rider_coor[0][0], rider_coor[0][1])

            self.buffer_rider.append(rider_coor[0][0])

            if len(self.buffer_rider) > self.buffer_rider_size_limit:
                self.buffer_rider = [self.buffer_rider[-1]]

            if len(tweezers_coor) == 1:
                tweezers_min_coordinate = np.array(tweezers_coor[0][0], tweezers_coor[0][1])
                # if rider move more than rider_move_threshold compared with first element stored in the buffer, rider consider moved
                if abs(rider_coor[0][0] - self.buffer_rider[0]) > self.rider_move_threshold:
                    self.buffer_rider = []  # buffer is cleared and store from [] again

                    # if rider move, check tweezers and rider distance
                    # if tweezers and rider are apart more than use_tweezers_threshold pixels (based on euclidean distance), consider not using tweezers
                    if np.linalg.norm(
                        rider_min_coordinate - tweezers_min_coordinate) <= self.use_tweezers_threshold and not self.rider_tweezers_lock_mark:
                        self.scoring['measuring_score_rider_tweezers'] = 1
                        self.keyframe['measuring_score_rider_tweezers'] = self.frame_counter

                    elif np.linalg.norm(rider_min_coordinate - tweezers_min_coordinate) > self.use_tweezers_threshold:
                        self.scoring['measuring_score_rider_tweezers'] = 0
                        self.keyframe['measuring_score_rider_tweezers'] = self.frame_counter
                        # once detected not using tweezers, will lose mark and not able to gain back this mark again
                        self.rider_tweezers_lock_mark = True
        # corner case: rider and tweezer can't found correctly at the same time.
        # we only detect tweezer_coor whether under balance
        elif len(tweezers_coor) == 1 and len(self.side_object_dict['balance']) == 1:
            tweezers_coor = self.side_object_dict['tweezers'][0]
            balance_coor = self.side_object_dict['balance'][0]
            if self.is_behind(tweezers_coor, balance_coor) and self.is_inside(self.get_center_coordinate(tweezers_coor),
                                                                              balance_coor):
                self.scoring['measuring_score_rider_tweezers'] = 1
                self.keyframe['measuring_score_rider_tweezers'] = self.frame_counter

    def evaluate_object_left(self):
        object_left_score, _ = self.evaluate_object_left_from_view(view='top')
        # 2-tray and 1-battery detected
        if object_left_score is not None and object_left_score >= 0:
            if (not self.is_object_put or self.is_change_object_direction) and object_left_score == 1:
                # give mark when object is put first time,
                # change mark only if student changes direction afterward
                self.is_change_object_direction = False
                self.is_object_put = True
                self.scoring['measuring_score_object_left'] = 1
                self.keyframe['measuring_score_object_left'] = self.frame_counter
            elif not self.is_object_put:
                self.keyframe['measuring_score_object_left'] = self.frame_counter
            elif self.is_object_put:
                self.scoring['measuring_score_object_left'] = object_left_score
                self.keyframe['measuring_score_object_left'] = self.frame_counter

    def evaluate_object_left_from_view(self, view):
        object_left_score = None
        object_left_keyframe = None

        # evaluate whether object(battery) is located at the left tray
        if view == 'side':
            tray_coor = self.side_object_dict['tray']
            balance_coor = self.side_object_dict['balance']
            battery_coors = self.side_object_dict['battery']
        elif view == 'top':
            tray_coor = self.top_object_dict['tray']
            balance_coor = self.top_object_dict['balance']
            battery_coors = self.top_object_dict['battery']

        if len(battery_coors) > 0 and len(tray_coor) == 2:
            # coordinate definition from COCO
            # (x_min,y_min)-------------+ -> x
            #             |             |
            #             |             |
            #             |             |
            #             +------------(x_max,y_max)
            #             y
            #
            # compare x_min (horizontal index) of two traies to locate [left_tray, right_tray]

            if tray_coor[0][0] <= tray_coor[1][0]:
                left_tray = tray_coor[0]
                right_tray = tray_coor[1]
            else:
                left_tray = tray_coor[1]
                right_tray = tray_coor[0]

            for battery_coor in battery_coors:
                battery_center_coor = self.get_center_coordinate(battery_coor)
                is_inside_left = self.is_inside(battery_center_coor, left_tray)
                is_inside_right = self.is_inside(battery_center_coor, right_tray)

                if not is_inside_left and not is_inside_right:
                    self.object_is_in_tray_now = False
                    if not self.is_object_put:
                        object_left_score = 0
                        object_left_keyframe = self.frame_counter
                elif is_inside_left:
                    self.object_direction = 'left'
                    self.object_is_in_tray_now = True
                    object_left_score = 1
                    object_left_keyframe = self.frame_counter
                    # no matter how many battery detected, as long as
                    # one object detected at left tray, consider get mark
                else:  # if object put happens on the right, then give zero score mark
                    self.object_direction = 'right'
                    self.object_is_in_tray_now = True
                    if not self.is_object_put or self.is_change_object_direction:
                        self.is_object_put = True
                        self.is_change_object_direction = False
                        object_left_score = 0
                        object_left_keyframe = self.frame_counter
                # if object is put at left initially, but change to right tray afterward, will reevaluate object_left mark
                if self.object_direction == 'left' and is_inside_right > 0:
                    self.is_change_object_direction = True
                if self.object_direction == 'right' and is_inside_left > 0:
                    self.is_change_object_direction = True
                return object_left_score, object_left_keyframe

        elif len(battery_coors) > 0 and len(balance_coor) == 1:
            # divide balance into 2 parts, left balance and right balance
            x_min, y_min, x_max, y_max = balance_coor[0]
            left_balance = x_min, y_min, x_min + (x_max - x_min) / 2, y_max
            right_balance = x_min + (x_max - x_min) / 2, y_min, x_max, y_max

            for battery_coor in battery_coors:
                battery_center_coor = self.get_center_coordinate(battery_coor)
                is_inside_left = self.is_inside(battery_center_coor, left_balance)
                is_inside_right = self.is_inside(battery_center_coor, right_balance)

                if not is_inside_left and not is_inside_right:
                    self.object_is_in_tray_now = False
                    if not self.is_object_put:
                        object_left_score = 0
                        object_left_keyframe = self.frame_counter
                elif is_inside_left:
                    self.object_direction = 'left'
                    self.object_is_in_tray_now = True
                    # give mark when object is put first time,
                    # change mark only if student change direction afterward
                    if not self.is_object_put or self.is_change_object_direction:
                        self.is_change_object_direction = False
                        object_left_score = 1
                        object_left_keyframe = self.frame_counter
                        # no matter how many battery detected, as long as one
                        # object detected at left tray, consider get mark
                        return object_left_score, object_left_keyframe
                else:
                    self.object_direction = 'right'
                    self.object_is_in_tray_now = True
                    if not self.is_object_put or self.is_change_object_direction:
                        self.is_object_put = True
                        self.is_change_object_direction = False
                        object_left_score = 0
                        object_left_keyframe = self.frame_counter

                # if object is put at left initially, but change to right tray afterward,
                # will reevaluate object_left mark
                if self.object_direction == 'left' and is_inside_right:
                    self.is_change_object_direction = True
                elif self.object_direction == 'right' and is_inside_left:
                    self.is_change_object_direction = True

        return object_left_score, object_left_keyframe

    def evaluate_weights_right(self):
        # evaluate whether weights is at right or left tray, then give mark to measuring_score_weights_right
        # if weights is increased, the measuring_score_balance and self.balance_persist duration will be reset.
        # combine with self.weight_order_helper() to evaluate weights order
        tray_coor = self.top_object_dict['tray']
        weights_obj_coor = self.top_object_dict['weights']

        if len(tray_coor) == 2 and len(weights_obj_coor) > 0:
            # compare x_min of 2 tray to locate [left_tray, right_tray] instead of [right_tray,left_tray]
            if tray_coor[0][0] > tray_coor[1][0]:
                left_tray = tray_coor[1]
                right_tray = tray_coor[0]
            else:
                left_tray = tray_coor[0]
                right_tray = tray_coor[1]

            weight_inside_right_tray = []  # weight_inside_right_tray = ['weight_5g','weight_10g',...]
            weight_inside_left_tray = []

            # if weights is inside left/right tray, store in respective list (weight_inside_left/right_tray)
            for weight in weights_obj_coor:
                weight_center_coor = self.get_center_coordinate(weight[1])
                is_inside_right = self.is_inside(weight_center_coor, right_tray)
                is_inside_left = self.is_inside(weight_center_coor, left_tray)

                if is_inside_right:
                    weight_inside_right_tray.append(weight)
                elif is_inside_left:
                    weight_inside_left_tray.append(weight)

            # mark will be given if users put the weight at right tray, mark will be kept constant except if users change tray (eg right to left tray)
            # once the first weight is put in left/right tray, self.is_weights_put become True to show weights have been put
            if not self.is_weights_put or self.is_change_weights_direction:
                if len(weight_inside_right_tray) > 0:
                    self.is_weights_put = True  # self.is_weights_put is prerequisites to enter end state
                    self.weights_direction = 'right'
                    self.is_change_weights_direction = False
                    self.scoring["measuring_score_weights_right"] = 1
                    self.keyframe["measuring_score_weights_right"] = self.frame_counter
                elif len(weight_inside_left_tray) > 0:
                    self.is_weights_put = True
                    self.weights_direction = 'left'
                    self.is_change_weights_direction = False
                    self.scoring['measuring_score_weights_right'] = 0
                    self.keyframe["measuring_score_weights_right"] = self.frame_counter

            # if user change the direction of weights, eg from right tray to left,
            # 'weights_right' mark will be re-evaluated
            if self.weights_direction == 'right' and len(weight_inside_left_tray) > 0:
                self.is_change_weights_direction = True
            elif self.weights_direction == 'left' and len(weight_inside_right_tray) > 0:
                self.is_change_weights_direction = True

            # check if students put/take weights using tweezers
            if self.weights_direction == 'right':
                self.evaluate_weights_tweezers(weight_inside_right_tray)
            elif self.weights_direction == 'left':
                self.evaluate_weights_tweezers(weight_inside_left_tray)

    def evaluate_weights_tweezers(self, weight_inside_tray):
        tweezers_coor = self.top_object_dict['tweezers']
        # if number of weights inside tray increase/decrease,
        # check relative position (euclidean distance) of tweezers and all weights (top_left)
        if self.num_weight_inside_tray != len(weight_inside_tray) and len(tweezers_coor) == 1:
            self.num_weight_inside_tray = len(weight_inside_tray)

            if not self.is_weight_tweezers_lock_mark:
                use_tweezers_bool = []
                for weight in weight_inside_tray:
                    a = np.array(weight[1][0], weight[1][1])
                    b = np.array(tweezers_coor[0][0], tweezers_coor[0][0])
                    if np.linalg.norm(a - b) <= self.use_tweezers_threshold:
                        use_tweezers_bool.append(True)
                    else:
                        use_tweezers_bool.append(False)

                if not self.tweezers_warning \
                    or self.frame_counter - self.tweezers_warning < self.tweezers_warning_duration:
                    self.tweezers_warning = None
                    if all(use_tweezers_bool) and len(use_tweezers_bool) > 0:
                        self.scoring['measuring_score_weights_tweezers'] = 1
                        self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter
                    else:
                        self.scoring['measuring_score_weights_tweezers'] = 0
                        self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter
                else:
                    self.is_weight_tweezers_lock_mark = True
            else:
                self.tweezers_warning = self.frame_counter
                self.scoring['measuring_score_weights_tweezers'] = 0
                self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter

    def evaluate_end_state(self):
        if self.is_object_put and self.is_weights_put:
            self.can_tidy = True

    def evaluate_end_tidy(self):
        # to get the tidy mark, students should keep all weights inside the box and clase the box,
        # put battery on the table, move rider back to zero position

        tray_coor = self.top_object_dict['tray']
        battery_coors = self.top_object_dict['battery']
        balance_coor = self.top_object_dict['balance']
        weights_obj_coor = self.top_object_dict['weights']

        if len(weights_obj_coor) == 0:
            if len(tray_coor) == 2 and len(battery_coors) > 0:
                for battery_coor in battery_coors:
                    # if battery is removed from the left or right tray, and rider is pushed to zero. get mark
                    battery_center_coor = self.get_center_coordinate(battery_coor)

                    if self.is_inside(small_item_center_coor=battery_center_coor, big_item_coor=tray_coor[0]) \
                        or self.is_inside(small_item_center_coor=battery_center_coor, big_item_coor=tray_coor[1]):

                        self.scoring["end_score_tidy"] = 0
                        self.keyframe["end_score_tidy"] = self.frame_counter
                    else:
                        self.evaluate_rider()
                        if self.rider_zero:
                            self.scoring["end_score_tidy"] = 1
                            self.keyframe["end_score_tidy"] = self.frame_counter
                        else:
                            self.scoring["end_score_tidy"] = 0
                            self.keyframe["end_score_tidy"] = self.frame_counter

            elif len(balance_coor) == 1 and len(battery_coors) > 0:
                # if all battery are removed from the left or right tray, and rider is pushed to zero. get mark
                for battery_coor in battery_coors:
                    battery_center_coor = self.get_center_coordinate(battery_coor)
                    # divide balance into 2 parts, left balance and right balance
                    x_min, y_min, x_max, y_max = balance_coor[0]
                    left_balance = x_min, y_min, x_min + (x_max - x_min) / 2, y_max
                    right_balance = x_min + (x_max - x_min) / 2, y_min, x_max, y_max

                    if self.is_inside(small_item_center_coor=battery_center_coor, big_item_coor=left_balance) \
                        or self.is_inside(small_item_center_coor=battery_center_coor, big_item_coor=right_balance):
                        self.scoring["end_score_tidy"] = 0
                        self.keyframe["end_score_tidy"] = self.frame_counter
                    else:
                        self.evaluate_rider()
                        if self.rider_zero:
                            self.scoring["end_score_tidy"] = 1
                            self.keyframe["end_score_tidy"] = self.frame_counter
                        else:
                            self.scoring["end_score_tidy"] = 0
                            self.keyframe["end_score_tidy"] = self.frame_counter

            elif len(battery_coors) == 0:
                self.evaluate_rider()
                if self.rider_zero:
                    self.scoring["end_score_tidy"] = 1
                    self.keyframe["end_score_tidy"] = self.frame_counter
        else:
            self.scoring["end_score_tidy"] = 0
            self.keyframe["end_score_tidy"] = self.frame_counter

    def get_balance_mark(self):
        if self.state == 'Initial':
            self.scoring['initial_score_balance'] = 1
            self.keyframe['initial_score_balance'] = self.frame_counter
        elif self.state == "Measuring" and self.is_object_put and self.scoring['end_score_tidy'] == 0:
            # double check object left in case 'end_score_tidy' not getting marks but object has been removed
            object_left_score, _ = self.evaluate_object_left_from_view(view='top')
            if object_left_score is not None and object_left_score >= 0:
                self.balance_persist_duration += 1
                if self.balance_persist_duration > self.balance_persist_duration_threshold:
                    self.scoring['measuring_score_balance'] = 1
                    self.keyframe['measuring_score_balance'] = self.frame_counter
                    self.measuring_state_balance_lock_mark = True

    def lose_balance_mark(self):
        if self.state == 'Initial':
            self.scoring['initial_score_balance'] = 0
            self.keyframe['initial_score_balance'] = self.frame_counter
        elif self.state == "Measuring":
            self.balance_persist_duration = 0
            if not self.measuring_state_balance_lock_mark:
                self.scoring['measuring_score_balance'] = 0
                self.keyframe['measuring_score_balance'] = self.frame_counter

    def evaluate_scale_balance(self):
        """
        In initial state, this function always running until first put_take action is detected
        In measuring state, after object is put, if the balance is balance more than self.balance_persist_duration_threshold frames, consider balance
        however, everytime weights are added to the balance, the mark and self.balance_persist_duration counter will be reset and start over again
        (defined in self.weight_order_helper function)
        """
        roundscrew2_coor = self.top_object_dict['roundscrew2']
        pointerhead_coor = self.top_object_dict['pointerhead']
        scale_coor = self.top_object_dict['scale']
        pointer_coor = self.top_object_dict['pointer']

        # if more than one pointerhead detected, estimate correct pointerhead from pointer coordinate (top center of pointer)
        if len(pointerhead_coor) > 1 and len(pointer_coor) == 1:
            estimate_coor = np.array([(pointer_coor[0][0] + pointer_coor[0][2]) / 2, pointer_coor[0][1]])
            distance = 10000000  # a random very large value
            for pointerhead in pointerhead_coor:
                candidate = np.array([(pointerhead[0] + pointerhead[2]) / 2, pointerhead[1] + pointerhead[3] / 2])
                if np.linalg.norm(estimate_coor - candidate) < distance:
                    best_candidate = pointerhead
                    distance = np.linalg.norm(estimate_coor - candidate)
            pointerhead_coor = [np.array(best_candidate)]

        # if no pointerhead detected, take top center of pointer as pointerhead
        if len(pointerhead_coor) == 0 and len(pointer_coor) == 1:
            x, y = (pointer_coor[0][0] + pointer_coor[0][2]) / 2, pointer_coor[0][1]
            pointerhead_coor = [np.array([x, y, x, y])]

        # only evaluate balance when 2 roundscrew2, 1 scale, 1 pointerhead and 1 pointer are found
        if len(roundscrew2_coor) == 2 and len(pointerhead_coor) == 1:
            # figure out left/right roundscrew2
            if roundscrew2_coor[0][0] < roundscrew2_coor[1][0]:
                left_roundscrew2_coor = roundscrew2_coor[0]
                right_roundscrew2_coor = roundscrew2_coor[1]
            else:
                left_roundscrew2_coor = roundscrew2_coor[1]
                right_roundscrew2_coor = roundscrew2_coor[0]

            # find center coordinate of roundscrew2 and pointerhead
            left_roundscrew2_center_coor = [(left_roundscrew2_coor[0] + left_roundscrew2_coor[2]) / 2,
                                            (left_roundscrew2_coor[1] + left_roundscrew2_coor[3]) / 2]
            right_roundscrew2_center_coor = [(right_roundscrew2_coor[0] + right_roundscrew2_coor[2]) / 2,
                                             (right_roundscrew2_coor[1] + right_roundscrew2_coor[3]) / 2]
            pointerhead_center_coor = [(pointerhead_coor[0][0] + pointerhead_coor[0][2]) / 2,
                                       (pointerhead_coor[0][1] + pointerhead_coor[0][3]) / 2]

            # rotate to make two roundscrew1 in a horizontal line
            rotated_left_coor, rotated_right_coor, rotated_center_coor = \
                self.rotate(left=left_roundscrew2_center_coor,
                            right=right_roundscrew2_center_coor,
                            center=pointerhead_center_coor)

            # if pointerhead center coordinate lies between [lower_limit,upper_limit], consider balance, where limit is middle point of two roundscrew2 +- balance_threshold
            lower_limit = (rotated_left_coor[0] + rotated_right_coor[0]) / 2 - self.balance_threshold
            upper_limit = (rotated_left_coor[0] + rotated_right_coor[0]) / 2 + self.balance_threshold

            if rotated_center_coor[0] < upper_limit and rotated_center_coor[0] > lower_limit:
                self.get_balance_mark()
            else:
                self.lose_balance_mark()

        elif len(scale_coor) == 1 and len(pointerhead_coor) == 1:
            pointerhead_center_coor = [(pointerhead_coor[0][0] + pointerhead_coor[0][2]) / 2,
                                       (pointerhead_coor[0][1] + pointerhead_coor[0][3]) / 2]
            lower_limit = (scale_coor[0][0] + scale_coor[0][2]) / 2 - self.balance_threshold
            upper_limit = (scale_coor[0][0] + scale_coor[0][2]) / 2 + self.balance_threshold

            if pointerhead_center_coor[0] < upper_limit and pointerhead_center_coor[0] > lower_limit:
                self.get_balance_mark()
            else:
                self.lose_balance_mark()

    def check_score_validity(self):
        for score_item, keyframe in self.keyframe.items():
            if keyframe == 0:
                self.scoring[score_item] = '-'
