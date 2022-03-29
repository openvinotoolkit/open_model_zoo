import math
import numpy as np


class Evaluator(object):
    def __init__(self):
        '''Score Evaluation Variables'''
        self.eval_vars = None
        self.eval_cbs = None
        self.top_scores_df = None
        self.video_eval_box = None
        self.front_scores_df = None

        self.action_mode = 'skip_frame' #action mode can be 'skip_frame' (for mobilenet) or 'batch_mode' (for mstcn)
        self.buffer_size = 40
        self.batch_mode_batchsize = 24
        self.batches_to_form_batchmode_buffer = 2
        self.buffer_pointer = 0
        self.filter_action_threshold = 15
        self.frame_top_buffer = []
        self.frame_side_buffer = []
        self.action_buffer = []
        self.action_pool_buffer = []
        self.pool_buffer_size = self.batch_mode_batchsize*self.batches_to_form_batchmode_buffer
        self.last_action = None
        self.top_obj_buffer = []
        self.front_obj_buffer = []
        self.frame_counter = 0
        self.top_object_dict = {}
        self.front_object_dict = {}
        self.first_put_take = False
        self.state = "Initial"
        self.buffer_rider = []    # buffer store coordinate of rider and tweezers to detect the move of rider, to evaluate the use of tweezers when adjust rider
        self.buffer_rider_size_limit = 30
        self.use_tweezers_threshold = 350   #200    # if tweezer and rider/weight distance more than tweezer treshold, consider use hand instead of use tweezer
        self.tweezers_warning = None
        self.tweezers_warning_duration = 60
        self.object_put = False # if battery is put on tray no matter left or right, return True
        self.object_direction = None # if initially object put in left, then change to right or vice versa, use this parameter to know the change to give mark again
        self.weights_put = False    # if weight is put on tray no matter left or right, return True
        self.weights_direction = None # if initially weights put in right, then change to left or vice versa, use this parameter to know the change to give mark again
        self.can_tidy = False # if battery and weight has been put on the tray(no matter left/right), then can tidy becomes true
        self.rider_move_threshold = 20 # if rider moves more than this value, check if tweezers or hand is used to move rider
        self.rider_zero=False
        self.rider_portion = 6  # divide the distance between 2 roundscrew1 into rider_portion portion, if rider falls in the first portion mean rider at zeroth position
        self.num_weight_inside_tray = 0 # use in weight order evaluation -- only evaluate order when there is changes in this parameter
        self.record_weight_inside_tray = [] # use in weight order evaluation -- only evaluate order when there is changes in this parameter
        self.balance_threshold = 12
        self.rider_tweezers_lock_mark = False
        self.weight_order_lock_mark = False
        self.weight_tweezers_lock_mark = False
        self.change_object_direction = False # mark given for object left will not change once given, except when this parameter become true 
        self.change_weights_direction = False
        self.balance_persist_duration = 0
        self.balance_persist_duration_threshold = 60
        self.measuring_state_balance_lock_mark = False  # once this variable is True, even the balance is not balanced, measuring state balance mark still is given. \
                                                        # it only turns true if balance persists more than self.balance_persist_duration_threshold frames
                                                        # however, when weights are added to the tray, this lock will become False again
        # scoring
        self.scoring = {
            "initial_score_rider":0,
            "initial_score_balance":0,
            "measuring_score_rider_tweezers":0,
            "measuring_score_balance":0,
            "measuring_score_object_left":0,
            "measuring_score_weights_right":0,
            "measuring_score_weights_tweezers":0,
            "measuring_score_weights_order":0,
            "end_score_tidy":0
        }

        # keyframe
        self.keyframe = {
            "initial_score_rider":0,
            "initial_score_balance":0,
            "measuring_score_rider_tweezers":0,
            "measuring_score_balance":0,
            "measuring_score_object_left":0,
            "measuring_score_weights_right":0,
            "measuring_score_weights_tweezers":0,
            "measuring_score_weights_order":0,
            "end_score_tidy":0
        }

    def inference(self,
        top_det_results,
        side_det_results,
        action_seg_results,
        frame_top,
        frame_side,
        frame_counter,
        mode='skip_frame'):
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

        action_seg_results = self.filter_action(action_seg_results, mode=self.action_mode)
        top_det_results,side_det_results = self.filter_object(top_det_results, side_det_results)
        frame_top,frame_side = self.filter_image(frame_top, frame_side)

        if frame_counter > self.pool_buffer_size:
            self.classify_state(action_seg_results)
            self.top_object_dict = self.get_object(det_results=top_det_results)
            self.front_object_dict = self.get_object(det_results=side_det_results)

            if self.state == "Initial":
                self.evaluate_rider()
                self.evaluate_scale_balance()

            elif self.state == "Measuring":
                if action_seg_results == "put_take" or action_seg_results == "put_left" or action_seg_results == "put_right" or \
                    action_seg_results == "take_left" or action_seg_results == "take_right":
                    self.evaluate_object_left()
                    self.evaluate_weights_right()
                    if not self.can_tidy:
                        self.evaluate_end_state()
                self.evaluate_scale_balance()

                if action_seg_results == "adjust_rider":
                    self.evaluate_rider_tweezers()
                    
                if self.object_put == True and self.weights_put == True:    # if object and weights are put no matter at left/right tray, 
                    self.evaluate_end_tidy()                                # then can start evaluate whether they keep the apparatus

            self.check_score_validity()
            display_frame_counter = frame_counter - self.pool_buffer_size

        else:
            display_frame_counter = 0
            action_seg_results = None

        return self.state, self.scoring, self.keyframe, action_seg_results, \
            top_det_results, side_det_results, frame_top, frame_side, display_frame_counter

    def check_consecutive(self):
        '''
        Given a list: ['noise','noise','noise','put_take','noise','noise','noise'], this function calculate number of consecutive element, store in count_list
        and return the first action that persist more than filter_action_threshold frame

        count_list = [('noise', 3), ('put_take', 1), ('adjust_rider', 3)]
        return 'noise' (as is the first action persist more than filter_action_threshold frame, assume threshold is 3)
        '''
        count_list = []
        count = 1
        for i in range(len(self.action_buffer)-1):

            if self.action_buffer[i] == self.action_buffer[i + 1]:
                count+=1
            elif self.action_buffer[i] is not self.action_buffer[i + 1]:
                count_list.append((self.action_buffer[i], count))
                count = 1
            if i == len(self.action_buffer) -2:
                count_list.append((self.action_buffer[i], count))
        for i in range(len(count_list)):
            if count_list[i][1] >= self.filter_action_threshold:
                return count_list[i][0]
            elif i == len(count_list)-1:
                return 'noise_action'

    def filter_action(self, action_seg_results, mode):
        """filter action segmentation result. action which persists less than filter_action_threshold frame will be ignored"""
        if mode == 'skip_frame':
            # at beginning of evaluation (less than self.buffer_size frame), action is None
            self.pool_buffer_size = self.buffer_size
            if len(self.action_buffer) < self.buffer_size:
                self.action_buffer.append(action_seg_results)
                return None
            # after self.buffer_size frame, filtering of action occurs
            elif len(self.action_buffer) == self.buffer_size:
                if self.action_buffer[0] is not self.last_action:
                    self.last_action = self.check_consecutive()
                self.action_buffer.pop(0)
                self.action_buffer.append(action_seg_results)
                return self.last_action
        elif mode == 'batch_mode':
            if action_seg_results != []:
                self.action_pool_buffer.extend(action_seg_results)
            if len(self.action_pool_buffer) > self.pool_buffer_size: # 48
                self.action_pool_buffer = self.action_pool_buffer[self.batch_mode_batchsize: ] # self.batch_mode_batchsize: 24
                self.buffer_pointer = 0

            if len(self.action_pool_buffer) < self.pool_buffer_size:
                return None
            elif len(self.action_pool_buffer) == self.pool_buffer_size:
                self.buffer_pointer += 1
                self.action_buffer = self.action_pool_buffer[self.buffer_pointer: self.buffer_pointer + self.batch_mode_batchsize]
                if self.action_buffer[0] is not self.last_action:
                    self.last_action = self.check_consecutive()

                return self.last_action

    def filter_object(self, top_det_results, side_det_results):
        if len(self.top_obj_buffer) < self.buffer_size:
            self.top_obj_buffer.append(top_det_results)
            self.front_obj_buffer.append(side_det_results)
            first_top_det_results = first_side_det_results = [np.array([[100, 500, 100, 500]]),[],np.array(['balance']),]
            return first_top_det_results, first_side_det_results

        # after self.buffer_size frame, filtering of action occurs
        elif len(self.top_obj_buffer) == self.buffer_size:
            first_top_det_results = self.top_obj_buffer.pop(0)
            first_side_det_results = self.front_obj_buffer.pop(0)
            self.top_obj_buffer.append(top_det_results)
            self.front_obj_buffer.append(side_det_results)

        return first_top_det_results, first_side_det_results

    def filter_image(self, frame_top, frame_side):
        if len(self.frame_top_buffer) < self.buffer_size:
            self.frame_top_buffer.append(frame_top)
            self.frame_side_buffer.append(frame_side)
            h, w, c = frame_top.shape
            blank_image = 255 * np.ones(shape=(h, w, c), dtype=np.uint8)
            first_frame_top = first_frame_side = blank_image
            return first_frame_top,first_frame_side

        # after self.buffer_size frame, filtering of action occurs
        elif len(self.frame_top_buffer) == self.buffer_size:
            first_frame_top = self.frame_top_buffer.pop(0)
            first_frame_side = self.frame_side_buffer.pop(0)
            self.frame_top_buffer.append(frame_top)
            self.frame_side_buffer.append(frame_side)

        return first_frame_top,first_frame_side

    def get_object(self, det_results):
        '''
        Most object return list of obj coordinate in dict except weight
        most object:
        return a dictionary of list of object coordinate, eg.[array([x_min,y_min,x_max,y_max]),...]
        
        weight:
        weights_obj_coor return list of object name and coordinate, eg.[[[weights5_g,array([x_min,y_min,x_max,y_max])],...]
        as weight (5g,10g) etc is important for weight_order algo
        '''

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

        for obj,coor in zip(det_results[2], det_results[0]):
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
                weights_obj_coor.append([obj,coor])
            elif obj == 'weight_10g':
                weights_obj_coor.append([obj,coor])
            elif obj == 'weight_20g':
                weights_obj_coor.append([obj,coor])
            elif obj == 'weight_50g':
                weights_obj_coor.append([obj,coor])
            elif obj == 'weight_100g':
                weights_obj_coor.append([obj,coor])
            elif obj == 'weights':
                weights_obj_coor.append([obj,coor])
            
        if self.state == 'Initial':
            i_object = {'rider': rider_coor, 'pointer': pointer_coor, 'pointerhead': pointerhead_coor, \
                        'roundscrew1': roundscrew1_coor, 'roundscrew2': roundscrew2_coor, \
                        'support_sleeve': support_sleeve_coor, 'pointer_sleeve': pointer_sleeve_coor}
            return i_object

        elif self.state == 'Measuring':
            m_object = {'rider': rider_coor, 'pointer': pointer_coor, 'pointerhead': pointerhead_coor, \
                        'roundscrew1': roundscrew1_coor, 'roundscrew2': roundscrew2_coor, 'battery': battery_coor, 'balance': balance_coor, \
                        'support_sleeve': support_sleeve_coor, 'pointer_sleeve': pointer_sleeve_coor, 'tray': tray_coor, \
                        'tweezers': tweezers_coor, 'weights': weights_obj_coor}
            return(m_object)

    def get_center_coordinate(self, coor):
        [x_min, y_min, x_max, y_max] = coor
        center_coor = (x_min + x_max) / 2, (y_min + y_max) / 2
        return center_coor

    def is_inside(self, small_item_center_coor, big_item_coor):
        [big_x_min, big_y_min, big_x_max, big_y_max] = big_item_coor
        [small_center_x, small_center_y] = small_item_center_coor
        if small_center_x >= big_x_min and small_center_x <= big_x_max and small_center_y > big_y_min and small_center_y < big_y_max:
            return True
        else:
            return False

    def classify_state(self, action_seg_results):
        if action_seg_results == "put_left" or action_seg_results == "put_right" or \
            action_seg_results == "take_left" or action_seg_results == "take_right" or \
            action_seg_results == 'put_take':       #TODO:  filter input data for action so that only action persist  more than certain frame taken as true data
            self.first_put_take = True
        if self.first_put_take == True: 
            self.state = "Measuring"
        elif self.first_put_take == False:
            self.state = "Initial"

    def rotate(self, left, right, center):
        '''
        Given 3 points, fix the left point, rotate until the right point is horizontal to the left point
        Return the rotated coordinate of the three points

        Used to correct angle in adjust_rider (roundscrew1 and rider) and balance(roundscrew2 and pointerhead)
        '''
        [left_x,left_y] = left
        [right_x,right_y] = right
        [center_x,center_y] = center

        # theta is the angle need to be rotated so that two roundscrew2 is horizontal
        if right_x - left_x is not 0: # to avoid num/zero = infinity
            if right_y <= left_y:
                theta = abs(math.atan((right_y - left_y) / (right_x - left_x)))   # angle in rad by default
            elif right_y > left_y:
                theta = -abs(math.atan((right_y - left_y) / (right_x - left_x)))
        else:
            theta = 0
        # offset to make left roundscrew coordinate as (0,0) so that we can use rotation matrix
        offset_x = left_x
        offset_y = left_y

        rotated_left_coor = int(((left_x - offset_x) * math.cos(theta) - (left_y - offset_y) * math.sin(theta)) + offset_x), \
                                        int(((left_x - offset_x) * math.sin(theta) + (left_y - offset_y) * math.cos(theta)) + offset_y)
        rotated_right_coor = int(((right_x - offset_x) * math.cos(theta) - (right_y-offset_y)*math.sin(theta)) + offset_x), \
                                        int(((right_x - offset_x) * math.sin(theta) + (right_y - offset_y) * math.cos(theta)) + offset_y)
        rotated_center_coor = int(((center_x-offset_x)*math.cos(theta) - (center_y-offset_y)*math.sin(theta)) + offset_x), \
                                        int(((center_x - offset_x) * math.sin(theta) + (center_y - offset_y) * math.cos(theta)) + offset_y)

        return rotated_left_coor, rotated_right_coor, rotated_center_coor

    def evaluate_rider(self):
        """
        Function:
            To evaluate whether rider is pushed to zero position
        """
        roundscrew1_coor = self.front_object_dict['roundscrew1']
        rider_coor = self.front_object_dict['rider']

        # only evaluate rider_zero if 2 roundscrew1 and 1 rider are found
        if len(roundscrew1_coor)== 2 and len(rider_coor) == 1:

            # find center coordinate of rider and roundscrew1
            (x0, y0) = \
                ((roundscrew1_coor[0][2] + roundscrew1_coor[0][0]) / 2,(roundscrew1_coor[0][3] + roundscrew1_coor[0][1]) / 2)
            (x1,y1) = \
                ((roundscrew1_coor[1][2] + roundscrew1_coor[1][0]) / 2,(roundscrew1_coor[1][3] + roundscrew1_coor[1][1]) / 2)
            rider_center_coor = \
                ((rider_coor[0][2] + rider_coor[0][0]) / 2,(rider_coor[0][3] + rider_coor[0][1]) / 2)

            # determine left/right roundscrew
            if x0 < x1:
                left_roundscrew1_center_coor = [x0, y0]
                right_roundscrew1_center_coor = [x1, y1]
            else:
                left_roundscrew1_center_coor = [x1, y1]
                right_roundscrew1_center_coor = [x0, y0]

            # rotate to make two roundscrew1 in a horizontal line
            rotated_left_coor, rotated_right_coor, rotated_center_coor = \
                self.rotate(left=left_roundscrew1_center_coor, right=right_roundscrew1_center_coor, center=rider_center_coor)
            limit = rotated_left_coor[0] + (rotated_right_coor[0] - rotated_left_coor[0]) / self.rider_portion

            # if rider center position < 1/10 of length between 2 roundscrew, consider rider is pushed to zero position
            if rotated_center_coor[0] < limit:
                if self.state=="Initial":
                    self.scoring["initial_score_rider"] = 1
                    self.keyframe["initial_score_rider"] = self.frame_counter
                elif self.state=='Measuring':
                    self.rider_zero=True    # self.rider_zero is to determine end_state score
            else:
                if self.state=="Initial":
                    self.scoring["initial_score_rider"] = 0
                    self.keyframe["initial_score_rider"] = self.frame_counter
                elif self.state=='Measuring':
                    self.rider_zero=False

    def evaluate_rider_tweezers(self):
        """
        Function:
            To evaluate whether rider is pushed using tweezers

        Logic:
            if rider moves, tweezers coordinate should within certain pixels (defined in self.use_tweezers_threshold) from the rider coordinate

        """
        rider_coor = self.front_object_dict['rider']
        tweezers_coor = self.front_object_dict['tweezers']

        # only evaluate rider_tweezers if 1 rider and 1 tweezers are found
        if len(rider_coor) == 1:
            rider_min_coordinate = np.array(rider_coor[0][0], rider_coor[0][1])

            self.buffer_rider.append(rider_coor[0][0])

            if len(self.buffer_rider) > self.buffer_rider_size_limit:
                self.buffer_rider = [self.buffer_rider[-1]]

            if len(tweezers_coor) == 1:
                tweezers_min_coordinate = np.array(tweezers_coor[0][0],tweezers_coor[0][1])
                # if rider move more than rider_move_threshold compared with first element stored in the buffer, rider consider moved
                if abs(rider_coor[0][0] - self.buffer_rider[0]) > self.rider_move_threshold:
                    self.buffer_rider = []    # buffer is cleared and store from [] again

                    # if rider move, check tweezers and rider distance
                    # if tweezers and rider are apart more than use_tweezers_threshold pixels (based on euclidean distance), consider not using tweezers
                    if np.linalg.norm(rider_min_coordinate - tweezers_min_coordinate) <= self.use_tweezers_threshold and self.rider_tweezers_lock_mark==False:
                        self.scoring['measuring_score_rider_tweezers'] = 1
                        self.keyframe['measuring_score_rider_tweezers'] = self.frame_counter

                    elif np.linalg.norm(rider_min_coordinate - tweezers_min_coordinate) > self.use_tweezers_threshold:
                        self.scoring['measuring_score_rider_tweezers'] = 0
                        self.keyframe['measuring_score_rider_tweezers'] = self.frame_counter
                        self.rider_tweezers_lock_mark = True # once detected not using tweezers, will lose mark and not able to gain back this mark again

    def evaluate_object_left(self):
        top_object_left_score, top_object_left_keyframe = self.evaluate_object_left_from_view(view='top')
        if top_object_left_score is not None:   # None when no 2 tray and 1 battery detected
            # if object is detected put at left tray from any view, give mark
            if (self.object_put == False or self.change_object_direction==True) and (top_object_left_score==1):
                self.object_put = True    # self.weights_put is prerequisites to enter end state
                self.scoring['measuring_score_object_left'] = 1
                self.keyframe['measuring_score_object_left'] = self.frame_counter
            elif self.object_put == False:
                self.keyframe['measuring_score_object_left'] = self.frame_counter

    def evaluate_object_left_from_view(self, view): #TODO: add balance as backup if tray not able to be detected
        object_left_score = None
        object_left_keyframe = None

        # to evaluate whether object(battery) is located at the left tray
        if view == 'front':
            battery_coors = self.front_object_dict['battery']
            tray_coor = self.front_object_dict['tray']
            balance_coor = self.front_object_dict['balance']
        elif view == 'top':
            battery_coors = self.top_object_dict['battery']
            tray_coor = self.top_object_dict['tray']
            balance_coor = self.top_object_dict['balance']

        if len(battery_coors) > 0 and len(tray_coor) == 2:
            #compare x_min of 2 tray to locate [left_tray, right_tray] instead of [right_tray,left_tray] 
            if tray_coor[0][0] >= tray_coor[1][0]:
                left_tray = tray_coor[1]
                right_tray = tray_coor[0]
                
            elif tray_coor[0][0] < tray_coor[1][0]:
                left_tray = tray_coor[0]
                right_tray = tray_coor[1]

            for battery_coor in battery_coors:
                battery_center_coor = self.get_center_coordinate(battery_coor)
                if self.is_inside(battery_center_coor, left_tray):
                    self.object_direction = 'left'
                    if self.object_put == False or self.change_object_direction:  # give mark when object is put first time, change mark only if student change direction afterward
                        self.change_object_direction = False
                        object_left_score = 1
                        object_left_keyframe = self.frame_counter
                        # no matter how many battery detected, as long as one object detected at left tray, consider get mark
                        return object_left_score, object_left_keyframe
                elif self.is_inside(battery_center_coor, right_tray):
                    self.object_direction = 'right'
                    if self.object_put == False or self.change_object_direction:
                        self.change_object_direction = False
                        self.object_put = True
                        object_left_score = 0
                        object_left_keyframe = self.frame_counter
                elif self.object_put == False:
                    object_left_score = 0
                    object_left_keyframe = self.frame_counter

                # if object is put at left initially, but change to right tray afterward, will reevaluate object_left mark
                if self.object_direction == 'left': 
                    if self.is_inside(battery_center_coor, right_tray) > 0:
                        self.change_object_direction = True
                elif self.object_direction == 'right':
                    if self.is_inside(battery_center_coor, left_tray) > 0:
                        self.change_object_direction = True

        elif len(battery_coors) > 0 and len(balance_coor) == 1:
            # divide balance into 2 parts, left balance and right balance
            x_min, y_min, x_max, y_max = balance_coor[0]
            left_balance = x_min, y_min, x_min + (x_max - x_min) / 2, y_max
            right_balance = x_min + (x_max - x_min) / 2, y_min, x_max, y_max

            for battery_coor in battery_coors:
                battery_center_coor = self.get_center_coordinate(battery_coor)
                if self.is_inside(battery_center_coor, left_balance):
                    self.object_direction = 'left'
                    if self.object_put == False or self.change_object_direction:  # give mark when object is put first time, change mark only if student change direction afterward
                        self.change_object_direction = False
                        object_left_score = 1
                        object_left_keyframe = self.frame_counter
                        # no matter how many battery detected, as long as one object detected at left tray, consider get mark
                        return object_left_score, object_left_keyframe
                elif self.is_inside(battery_center_coor, right_balance):
                    self.object_direction = 'right'
                    if self.object_put == False or self.change_object_direction:
                        self.change_object_direction = False
                        self.object_put = True
                        object_left_score = 0
                        object_left_keyframe = self.frame_counter
                elif self.object_put == False:
                    object_left_score = 0
                    object_left_keyframe = self.frame_counter

                # if object is put at left initially, but change to right tray afterward, will reevaluate object_left mark
                if self.object_direction == 'left': 
                    if self.is_inside(battery_center_coor, right_balance) > 0:
                        self.change_object_direction = True

                elif self.object_direction == 'right':
                    if self.is_inside(battery_center_coor, left_balance) > 0:
                        self.change_object_direction = True

        return object_left_score, object_left_keyframe

    def evaluate_weights_right(self):
        # evaluate whether weights is at right or left tray, then give mark to measuring_score_weights_right
        # if weights is increased, the measuring_score_balance and self.balance_persist duration will be reset.
        # combine with self.weight_order_helper() to evaluate weights order

        tray_coor = self.top_object_dict['tray']
        weights_obj_coor = self.top_object_dict['weights']

        if len(tray_coor) == 2 and len(weights_obj_coor) > 0:

            #compare x_min of 2 tray to locate [left_tray, right_tray] instead of [right_tray,left_tray] 
            if tray_coor[0][0] > tray_coor[1][0]:
                left_tray = tray_coor[1]
                right_tray = tray_coor[0]
                
            else:
                left_tray = tray_coor[0]
                right_tray = tray_coor[1]

            weight_inside_right_tray = []   # weight_inside_right_tray = ['weight_5g','weight_10g',...]
            weight_inside_left_tray = []

            # if weights is inside left/right tray, store in respective list (weight_inside_left/right_tray)
            for weight in weights_obj_coor:
                weight_center_coor = self.get_center_coordinate(weight[1])
                if self.is_inside(weight_center_coor, right_tray):
                    weight_inside_right_tray.append(weight[0])
                elif self.is_inside(weight_center_coor, left_tray):
                    weight_inside_left_tray.append(weight[0])

            # mark will be given if users put the weight at right tray, mark will be kept constant except if users change tray (eg right to left tray)
            # once the first weight is put in left/right tray, self.weights_put become True to show weights have been put
            if self.weights_put == False or self.change_weights_direction == True:
                if len(weight_inside_right_tray)>0:
                    self.scoring["measuring_score_weights_right"] = 1
                    self.keyframe["measuring_score_weights_right"] = self.frame_counter
                    self.weights_put = True    # self.weights_put is prerequisites to enter end state
                    self.change_weights_direction = False
                    self.weights_direction = 'right'
                elif len(weight_inside_left_tray) > 0:
                    self.scoring['measuring_score_weights_right'] = 0
                    self.keyframe["measuring_score_weights_right"] = self.frame_counter
                    self.weights_put = True
                    self.change_weights_direction = False
                    self.weights_direction = 'left'

            # if user change the direction of weights, eg from right tray to left, 'weights_right' mark will be re-evaluated
            if self.weights_direction == 'right':
                if len(weight_inside_left_tray) > 0:
                    self.change_weights_direction = True

            elif self.weights_direction == 'left':
                if len(weight_inside_right_tray) > 0:
                    self.change_weights_direction = True

            # check if students put/take weights using tweezers
            if self.weights_direction == 'right':
                self.evaluate_weights_tweezers(num_weight_inside_tray=len(weight_inside_right_tray))
            elif self.weights_direction == 'left':
                self.evaluate_weights_tweezers(num_weight_inside_tray=len(weight_inside_right_tray))

    def evaluate_weights_tweezers(self,num_weight_inside_tray):
        tweezers_coor = self.top_object_dict['tweezers']
        weights_coor = self.top_object_dict['weights']
        # if number of weights inside tray increase/decrease, check relative position (euclidean distance) of tweezers and all weights (top_left) 
        if self.num_weight_inside_tray != num_weight_inside_tray:
            self.num_weight_inside_tray = num_weight_inside_tray

            if len(tweezers_coor) == 1:
                use_tweezers_bool = []
                for weight_coor in self.top_object_dict['weights']:
                    if self.weight_tweezers_lock_mark==False:
                        weights_coor = self.top_object_dict['weights']
                        a = np.array(weight_coor[1][0], weight_coor[1][1])
                        b = np.array(tweezers_coor[0][0], tweezers_coor[0][0])
                        if np.linalg.norm(a-b) <= self.use_tweezers_threshold:
                            use_tweezers_bool.append(True)
                        else:
                            use_tweezers_bool.append(False)

                if any(use_tweezers_bool) and self.weight_tweezers_lock_mark==False:
                    if self.tweezers_warning == None or self.frame_counter-self.tweezers_warning < self.tweezers_warning_duration:
                        self.tweezers_warning = None
                        self.scoring['measuring_score_weights_tweezers'] = 1
                        self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter
                        for weight_coor in self.top_object_dict['weights']:
                            a = np.array(weight_coor[1][0], weight_coor[1][1])
                            b = np.array(tweezers_coor[0][0], tweezers_coor[0][0])
                    else:
                        self.weight_tweezers_lock_mark = True
                else:
                    print(f'gg: tweezers:{tweezers_coor} weights:{weights_coor}')
                    for weight_coor in self.top_object_dict['weights']:
                        a = np.array(weight_coor[1][0], weight_coor[1][1])
                        b = np.array(tweezers_coor[0][0], tweezers_coor[0][0])
                    self.scoring['measuring_score_weights_tweezers'] = 0
                    # lock at frame(self.frame_counter)
                    self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter
                    self.tweezers_warning = self.frame_counter

    def evaluate_end_state(self):
        if self.object_put == True and self.weights_put == True:
            self.can_tidy = True

    def evaluate_end_tidy(self):
        # to get the tidy mark, students should keep all weights inside the box and clase the box,
        # put battery on the table, move rider back to zero position

        tray_coor = self.top_object_dict['tray']
        battery_coors = self.top_object_dict['battery']
        balance_coor = self.top_object_dict['balance']
        weights_obj_coor = self.top_object_dict['weights']

        if len(weights_obj_coor) == 0:
            if len(tray_coor) == 2 and len(battery_coors)  > 0:
                for battery_coor in battery_coors:
                    # if battery is removed from the left or right tray, and rider is pushed to zero. get mark
                    battery_center_coor = self.get_center_coordinate(battery_coor)

                    if self.is_inside(small_item_center_coor=battery_center_coor, big_item_coor=tray_coor[0]) \
                        or self.is_inside(small_item_center_coor=battery_center_coor, big_item_coor=tray_coor[1]):

                        self.scoring["end_score_tidy"] = 0
                        self.keyframe["end_score_tidy"] = self.frame_counter 
                    else:
                        self.evaluate_rider()
                        if self.rider_zero == True:
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
                        if self.rider_zero == True:
                            self.scoring["end_score_tidy"] = 1
                            self.keyframe["end_score_tidy"] = self.frame_counter
                        else:
                            self.scoring["end_score_tidy"] = 0
                            self.keyframe["end_score_tidy"] = self.frame_counter
            elif len(battery_coors) == 0:
                self.evaluate_rider()
                if self.rider_zero == True:
                    self.scoring["end_score_tidy"] = 1
                    self.keyframe["end_score_tidy"] = self.frame_counter
        else:
            self.scoring["end_score_tidy"] = 0
            self.keyframe["end_score_tidy"] = self.frame_counter

    def evaluate_scale_balance(self):
        '''
        In initial state, this function always running until first put_take action is detected
        In measuring state, after object is put, if the balance is balance more than self.balance_persist_duration_threshold frames, consider balance
                            however, everytime weights are added to the balance, the mark and self.balance_persist_duration counter will be reset and start over again
                            (defined in self.weight_order_helper function)
        '''
        roundscrew2_coor = self.top_object_dict['roundscrew2']
        pointerhead_coor = self.top_object_dict['pointerhead']

        # only evaluate balance when 2 roundscrew2, 1 scale, 1 pointerhead and 1 pointer are found
        if len(roundscrew2_coor) == 2 and len(pointerhead_coor)==1:
            # figure out left/right roundscrew2  
            if roundscrew2_coor[0][0] < roundscrew2_coor[1][0]:
                left_roundscrew2_coor = roundscrew2_coor[0]
                right_roundscrew2_coor = roundscrew2_coor[1]
            else:
                left_roundscrew2_coor = roundscrew2_coor[1]
                right_roundscrew2_coor = roundscrew2_coor[0]

            # find center coordinate of roundscrew2 and pointerhead
            left_roundscrew2_center_coor = [(left_roundscrew2_coor[0] + left_roundscrew2_coor[2])/2, \
                                            (left_roundscrew2_coor[1] + left_roundscrew2_coor[3])/2]
            right_roundscrew2_center_coor = [(right_roundscrew2_coor[0] + right_roundscrew2_coor[2])/2, \
                                            (right_roundscrew2_coor[1] + right_roundscrew2_coor[3])/2]
            pointerhead_center_coor = [(pointerhead_coor[0][0] + pointerhead_coor[0][2])/2,\
                                            (pointerhead_coor[0][1] + pointerhead_coor[0][3])/2]

            # rotate to make two roundscrew1 in a horizontal line
            rotated_left_coor,rotated_right_coor,rotated_center_coor = \
                self.rotate(left=left_roundscrew2_center_coor,
                right=right_roundscrew2_center_coor,
                center=pointerhead_center_coor)

            # if pointerhead center coordinate lies between [lower_limit,upper_limit], consider balance, where limit is middle point of two roundscrew2 +- balance_threshold
            lower_limit = (rotated_left_coor[0] + rotated_right_coor[0]) / 2 - self.balance_threshold
            upper_limit = (rotated_left_coor[0] + rotated_right_coor[0]) / 2 + self.balance_threshold

            if rotated_center_coor[0] < upper_limit and rotated_center_coor[0] > lower_limit:
                if self.state == 'Initial':
                    self.scoring['initial_score_balance'] = 1
                    self.keyframe['initial_score_balance'] = self.frame_counter
                elif self.state == "Measuring" and self.object_put == True and self.scoring['end_score_tidy'] == 0:
                    self.balance_persist_duration += 1
                    if self.balance_persist_duration > self.balance_persist_duration_threshold:
                        self.scoring['measuring_score_balance'] = 1
                        self.keyframe['measuring_score_balance'] = self.frame_counter
                        self.measuring_state_balance_lock_mark = True
            else:
                if self.state == 'Initial':
                    self.scoring['initial_score_balance'] = 0
                    self.keyframe['initial_score_balance'] = self.frame_counter
                elif self.state == "Measuring":
                    self.balance_persist_duration = 0
                    if self.measuring_state_balance_lock_mark == False:
                        self.scoring['measuring_score_balance'] = 0
                        self.keyframe['measuring_score_balance'] = self.frame_counter

    def check_score_validity(self):
        for score_item,keyframe in self.keyframe.items():
            if keyframe==0:
                self.scoring[score_item] = '-'
