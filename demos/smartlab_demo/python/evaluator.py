# -- coding: utf-8 --
import math
import re

class Evaluator:
    def __init__(self):
        '''Score Evaluation Variables'''
        self.frame_counter = 0
        self.top_object_dict = {}
        self.side_object_dict = {}
        self.first_put_take = False
        self.state = "Initial"
        self.buffer_rider = []    # buffer store coordinate of rider and tweezers to detect the move of rider, to evaluate the use of tweezers when adjust rider
        self.use_tweezers_threshold = 350   #200    # if tweezer and rider/weight distance more than tweezer treshold, consider use hand instead of use tweezer
        self.object_put = False # if battery is put on tray no matter left or right, return True
        self.object_direction = None # if initially object put in left, then change to right or vice versa, use this parameter to know the change to give mark again
        self.weights_put = False    # if weight is put on tray no matter left or right, return True
        self.weights_direction = None # if initially weights put in right, then change to left or vice versa, use this parameter to know the change to give mark again
        self.can_tidy = False # if battery and weight has been put on the tray(no matter left/right), then can tidy becomes true
        self.rider_move_threshold = 8 # if rider moves more than this value, check if tweezers or hand is used to move rider
        self.rider_zero=False
        self.num_weight_inside_tray = 0 # use in weight order evaluation -- only evaluate order when there is changes in this parameter
        self.record_weight_inside_tray = [] # use in weight order evaluation -- only evaluate order when there is changes in this parameter
        self.balance_threshold = 5
        self.rider_tweezers_lock_mark = False
        self.weight_order_lock_mark = False
        self.weight_tweezers_lock_mark = False
        self.change_object_direction = False # mark given for object left will not change once given, except when this parameter become true 
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
        frame_top, frame_side,
        frame_counter):
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
        self.classify_state(action_seg_results)

        self.top_object_dict = self.get_object(det_results = top_det_results)
        self.side_object_dict = self.get_object(det_results = side_det_results)

        if self.state == "Initial":
            self.evaluate_rider()
            self.evaluate_scale_balance()

        elif self.state == "Measuring":
            
            if action_seg_results == "put_take" or action_seg_results == "put_left" or action_seg_results == "put_right" or \
                action_seg_results == "take_left" or action_seg_results == "take_right":
                self.evaluate_object_left()
                self.evaluate_weights_order()
                if not self.can_tidy:
                    self.evaluate_end_state()
            
            self.evaluate_scale_balance()

            if action_seg_results == "adjust_rider":
                self.evaluate_rider_tweezers()

            if self.object_put == True:     # temporary solution while waiting for object detection for weight to complete, remove requirement of weights to enter end_state evaluation 
                self.evaluate_end_tidy()

        self.check_score_validity()

        return self.state, self.scoring, self.keyframe

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

        if self.state == 'Initial':
            i_object = {'rider': rider_coor,'pointer':pointer_coor,'pointerhead': pointerhead_coor,
                        'roundscrew1': roundscrew1_coor,'roundscrew2': roundscrew2_coor,
                        'support_sleeve': support_sleeve_coor,'pointer_sleeve': pointer_sleeve_coor}

            return i_object

        elif self.state == 'Measuring':
            m_object = {'rider': rider_coor, 'pointer': pointer_coor, 'pointerhead': pointerhead_coor,
                        'roundscrew1': roundscrew1_coor, 'roundscrew2': roundscrew2_coor, 'battery': battery_coor, 'balance': balance_coor,
                        'support_sleeve': support_sleeve_coor, 'pointer_sleeve': pointer_sleeve_coor, 'tray': tray_coor,
                        'tweezers': tweezers_coor, 'weights': weights_obj_coor}

            return (m_object)

    def get_center_coordinate(self, coor):
        [x_min, y_min, x_max, y_max] = coor
        center_coor = (x_min + x_max) / 2, (y_min + y_max) / 2

        return center_coor

    def evaluate_weights_order(self):
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

            weight_inside_right_tray = [] # weight_inside_right_tray = ['weight_5g','weight_10g',...]
            weight_inside_left_tray = []

            # if weights is inside left/right tray, store in respective list (weight_inside_left/right_tray)
            for weight in weights_obj_coor:
                weight_center_coor = self.get_center_coordinate(weight[1])
                if self.is_inside(weight_center_coor,right_tray):
                    weight_inside_right_tray.append(weight[0])

                elif self.is_inside(weight_center_coor,left_tray):
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
                    
                elif len(weight_inside_left_tray)>0:
                    self.scoring['measuring_score_weights_right'] = 0
                    self.keyframe["measuring_score_weights_right"] = self.counter
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

            # check weight order for both tray, ie if students put all weights in left tray will correct order, will get marks also
            if self.weights_direction == 'right':
                self.weight_order_helper(weight_inside_right_tray)
            elif self.weights_direction == 'left':
                self.weight_order_helper(weight_inside_left_tray)

    def is_inside(self, small_item_center_coor, big_item_coor):
        [big_x_min, big_y_min, big_x_max, big_y_max] = big_item_coor
        [small_center_x, small_center_y] = small_item_center_coor
        if small_center_x >= big_x_min and small_center_x <= big_x_max and small_center_y > big_y_min and small_center_y < big_y_max:
            return True
        else:
            return False

    def weight_order_helper(self, weight_inside_tray):
        '''
        This function helps to identify which weights are added to or removed from the tray.
        From the addition or removal of weights, we can identify whether the order of addition/removal of weights is correct.
        At the moment when weights are removed or added to the tray, the use of tweezers is evaluated as well during the measuring state.
        '''

        if self.num_weight_inside_tray < len(weight_inside_tray): # weight is increased
            self.scoring['measuring_score_balance'] = 0
            self.keyframe['measuring_score_balance'] = self.frame_counter
            self.balance_persist_duration = 0
            self.measuring_state_balance_lock_mark = False

            newly_added = set(weight_inside_tray)-set(self.record_weight_inside_tray)
            if len(newly_added) == 0:
                newly_added = 0
            else:
                for x in newly_added:
                    newly_added = int(re.findall(r"\d+", x)[0])

            self.record_weight_inside_tray = weight_inside_tray
            
            if len(self.record_weight_inside_tray) > 0 and newly_added:
                old_weights = list(map(lambda x: int(re.findall(r"\d+", x)[0]), self.record_weight_inside_tray))
                if any(old_weight<newly_added for old_weight in old_weights):
                    self.scoring['measuring_score_weights_order'] = 0
                    self.keyframe['measuring_score_weights_order'] = self.frame_counter
                else:
                    self.scoring['measuring_score_weights_order'] = 1
                    self.keyframe['measuring_score_weights_order'] = self.frame_counter

            if newly_added:
                self.evaluate_weight_tweezers(weight_num = newly_added) # evaluate whether use tweezers/hand to move weight when there is change in number of weight inside tray

        elif self.num_weight_inside_tray > len(weight_inside_tray):    # weight is removed
            newly_removed = set(self.record_weight_inside_tray) - set(weight_inside_tray)
            if len(newly_removed) == 0:
                newly_removed = 0
            else:
                for x in newly_removed:
                    newly_removed = int(re.findall(r"\d+", x)[0])
            if len(self.record_weight_inside_tray) >0 and newly_removed:
                old_weights = list(map(lambda x: int(re.findall(r"\d+", x)[0]), self.record_weight_inside_tray))
                if any(old_weight >= newly_removed for old_weight in old_weights) and self.weight_order_lock_mark == False:
                    self.scoring["measuring_score_weights_order"] = 1
                    self.keyframe["measuring_score_weights_order"] = self.frame_counter

                elif any(old_weight<newly_removed for old_weight in old_weights):
                    self.scoring["measuring_score_weights_order"] = 0
                    self.keyframe["measuring_score_weights_order"] = self.frame_counter
                    self.weight_order_lock_mark == True     # once self.weight_order_lock_mark turns True, user not able to get this mark anymore

            self.record_weight_inside_tray = weight_inside_tray

            if newly_removed:
                self.evaluate_weight_tweezers(weight_num=newly_removed)     # evaluate whether use tweezers/hand to move weight when there is change in number of weight inside tray
        
        self.num_weight_inside_tray = len(weight_inside_tray)   # update num of weight in tray

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
        if right_x-left_x is not 0: # to avoid num/zero = infinity
            if right_y <= left_y:
                theta = abs(math.atan((right_y - left_y)/(right_x - left_x)))   # angle in rad by default
            elif right_y > left_y:
                theta = -abs(math.atan((right_y - left_y)/(right_x - left_x)))
        else:
            theta = 0
        # offset to make left roundscrew coordinate as (0,0) so that we can use rotation matrix
        offset_x = left_x
        offset_y = left_y

        rotated_left_coor = \
            int(((left_x - offset_x) * math.cos(theta) - (left_y - offset_y) * math.sin(theta)) + offset_x), \
            int(((left_x - offset_x) * math.sin(theta) + (left_y - offset_y) * math.cos(theta)) + offset_y)
        rotated_right_coor = \
            int(((right_x - offset_x) * math.cos(theta) - (right_y - offset_y) * math.sin(theta)) + offset_x), \
            int(((right_x - offset_x) * math.sin(theta) + (right_y - offset_y) * math.cos(theta)) + offset_y)
        rotated_center_coor = \
            int(((center_x - offset_x) * math.cos(theta) - (center_y - offset_y) * math.sin(theta)) + offset_x), \
            int(((center_x - offset_x) * math.sin(theta) + (center_y - offset_y) * math.cos(theta)) + offset_y)

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
            x0, y0 = \
                ((roundscrew1_coor[0][2] + roundscrew1_coor[0][0]) / 2, (roundscrew1_coor[0][3] + roundscrew1_coor[0][1]) / 2)
            (x1,y1) = \
                ((roundscrew1_coor[1][2] + roundscrew1_coor[1][0]) / 2, (roundscrew1_coor[1][3] + roundscrew1_coor[1][1]) / 2)
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
            rotated_left_coor, rotated_right_coor, rotated_center_coor = self.rotate( \
                left = left_roundscrew1_center_coor, right = right_roundscrew1_center_coor, center = rider_center_coor)
            limit = rotated_left_coor[0] + (rotated_right_coor[0] - rotated_left_coor[0]) / 8

            # if rider center position < 1/10 of length between 2 roundscrew, consider rider is pushed to zero position
            if rotated_center_coor[0] < limit:
                if self.state == "Initial":
                    self.scoring["initial_score_rider"] = 1
                    self.keyframe["initial_score_rider"] = self.frame_counter
                elif self.state == 'Measuring':
                    self.rider_zero=True # self.rider_zero is to determine end_state score
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
        if len(rider_coor) == 1 and len(tweezers_coor) == 1:
            rider_min_coordinate = \
                (rider_coor[0][0], rider_coor[0][1])
            tweezers_min_coordinate = \
                (tweezers_coor[0][0], tweezers_coor[0][1])
            self.buffer_rider.append(rider_min_coordinate[0])

            # if rider move more than rider_move_threshold compared with first element stored in the buffer, rider consider moved
            if abs(rider_min_coordinate[0] - self.buffer_rider[0]) > self.rider_move_threshold:
                self.buffer_rider = [] # buffer is cleared and store from [] again

                # if rider move, check tweezers and rider distance
                # if tweezers and rider are apart more than use_tweezers_threshold pixels (based on x-coordinate only), consider not using tweezers
                if abs(rider_min_coordinate[0] - tweezers_min_coordinate[0]) < self.use_tweezers_threshold \
                    and self.rider_tweezers_lock_mark == False:
                    self.scoring['measuring_score_rider_tweezers'] = 1
                    self.keyframe['measuring_score_rider_tweezers'] = self.frame_counter

                elif abs(rider_min_coordinate[0] - tweezers_min_coordinate[0]) > self.use_tweezers_threshold:
                    self.scoring['measuring_score_rider_tweezers'] = 0
                    self.keyframe['measuring_score_rider_tweezers'] = self.frame_counter
                    self.rider_tweezers_lock_mark = True # once detected not using tweezers, will lose mark and not able to gain back this mark again


    def evaluate_object_left(self):
        top_object_left_score, top_object_left_keyframe = \
            self.evaluate_object_left_from_view(view = 'top')
        if top_object_left_score is not None: # None when no 2 tray and 1 battery detected
            # if object is detected put at left tray from any view, give mark
            if (self.object_put == False or self.change_object_direction == True) \
                and (top_object_left_score == 1):

                self.object_put = True # self.weights_put is prerequisites to enter end state
                self.scoring['measuring_score_object_left'] = 1
                self.keyframe['measuring_score_object_left'] = self.frame_counter
            elif self.object_put == False:
                self.keyframe['measuring_score_object_left'] = self.frame_counter

    def evaluate_object_left_from_view(self, view):
        object_left_score=None
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
                if self.is_inside(battery_center_coor,left_tray):
                    self.object_direction = 'left'
                    # give mark when object is put first time, change mark only if student change direction afterward
                    if self.object_put == False or self.change_object_direction:
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
                    if self.is_inside(battery_center_coor, left_tray)>0:
                        self.change_object_direction = True

        elif len(battery_coors) > 0 and len(balance_coor) == 1:
            # divide balance into 2 parts, left balance and right balance
            x_min,y_min,x_max,y_max = balance_coor[0]
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
                    # print('battery at right')
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

        return object_left_score,object_left_keyframe

    def evaluate_weight_tweezers(self, weight_num): #TODO:  bug: when newly added/removed is 20g (we have 2 of it), it will not check this tweezers
        tweezers_coor = self.top_object_dict['tweezers']

        if len(tweezers_coor) == 1:
            weight_name = f'weight_{weight_num}g'
            if weight_name == 'weight_20g':
                weight_20g_use_tweezers = 0
                for obj,weight_coor in self.top_object_dict['weights']:
                    if abs(weight_coor[0] - tweezers_coor[0][0]) < self.use_tweezers_threshold \
                        and self.weight_tweezers_lock_mark == False:
                        weight_20g_use_tweezers += 1
                if weight_20g_use_tweezers > 0:
                    self.scoring['measuring_score_weights_tweezers'] = 1
                    self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter
                else:
                    self.scoring['measuring_score_weights_tweezers'] = 0
                    self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter
                    self.weight_tweezers_lock_mark = True
            else:
                for obj,weight_coor in self.top_object_dict['weights']:
                    if obj == weight_name:
                        if abs(weight_coor[0] - tweezers_coor[0][0]) < self.use_tweezers_threshold \
                            and self.weight_tweezers_lock_mark == False:
                            self.scoring['measuring_score_weights_tweezers'] = 1
                            self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter
                        else:
                            self.scoring['measuring_score_weights_tweezers'] = 0
                            self.keyframe['measuring_score_weights_tweezers'] = self.frame_counter
                            self.weight_tweezers_lock_mark = True

    def evaluate_end_state(self):
        if self.object_put == True and self.weights_put == True:
            self.can_tidy = True

    # top_det_results.obj.values -> temporal_segmentation class
    def evaluate_end_tidy(self):
        # to get the tidy mark, students should keep all weights inside the box and clase the box,
        # put battery on the table, move rider back to zero position
        weights_obj_coor = self.top_object_dict['weights']
        tray_coor = self.top_object_dict['tray']
        battery_coor = self.top_object_dict['battery']

        if len(weights_obj_coor) == 0:  # all weights have to be kept inside the box and close the box, so no weights should be detected
            if len(tray_coor) == 2 and len(battery_coor) == 1:
                # if battery is removed from the left or right tray, and rider is pushed to zero. get mark
                battery_center_coor = self.get_center_coordinate(battery_coor[0])
                if not self.is_inside(small_item_center_coor = battery_center_coor, big_item_coor = tray_coor[0]) \
                    or not self.is_inside(small_item_center_coor = battery_center_coor, big_item_coor = tray_coor[1]):

                    self.evaluate_rider()
                    if self.rider_zero == True:
                        self.scoring["end_score_tidy"] = 1
                        self.keyframe["end_score_tidy"] = self.frame_counter
                    else:
                        pass
                else:
                    self.scoring["end_score_tidy"] = 0
                    self.keyframe["end_score_tidy"] = self.frame_counter

    def evaluate_end_state(self):
        if self.object_put == True:
            self.can_tidy = True

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
        if len(roundscrew2_coor) == 2 and len(pointerhead_coor) == 1:
            # figure out left/right roundscrew2
            if roundscrew2_coor[0][0] < roundscrew2_coor[1][0]:
                left_roundscrew2_coor = roundscrew2_coor[0]
                right_roundscrew2_coor = roundscrew2_coor[1]
            else:
                left_roundscrew2_coor = roundscrew2_coor[1]
                right_roundscrew2_coor = roundscrew2_coor[0]

            # find center coordinate of roundscrew2 and pointerhead
            left_roundscrew2_center_coor = [ \
                (left_roundscrew2_coor[0] + left_roundscrew2_coor[2]) / 2,
                (left_roundscrew2_coor[1] + left_roundscrew2_coor[3]) / 2]
            right_roundscrew2_center_coor = [ \
                (right_roundscrew2_coor[0] + right_roundscrew2_coor[2]) / 2,
                (right_roundscrew2_coor[1] + right_roundscrew2_coor[3]) / 2]
            pointerhead_center_coor = [ \
                (pointerhead_coor[0][0] + pointerhead_coor[0][2]) / 2,
                (pointerhead_coor[0][1] + pointerhead_coor[0][3]) / 2]

            # rotate to make two roundscrew1 in a horizontal line
            rotated_left_coor, rotated_right_coor, rotated_center_coor = \
                self.rotate(left = left_roundscrew2_center_coor,
                right = right_roundscrew2_center_coor, center = pointerhead_center_coor)

            # if pointerhead center coordinate lies between [lower_limit,upper_limit], consider balance, where limit is middle point of two roundscrew2 +- balance_threshold
            lower_limit = (rotated_left_coor[0] + rotated_right_coor[0]) / 2 - self.balance_threshold
            upper_limit = (rotated_left_coor[0] + rotated_right_coor[0]) / 2 + self.balance_threshold

            if rotated_center_coor[0] < upper_limit and rotated_center_coor[0] > lower_limit:
                if self.state == 'Initial':
                    self.scoring['initial_score_balance'] = 1
                    self.keyframe['initial_score_balance'] = self.frame_counter
                elif self.state == "Measuring" and self.object_put == True \
                    and self.scoring['end_score_tidy'] == 0:
                    
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
        for score_item, keyframe in self.keyframe.items():
            if keyframe == 0:
                self.scoring[score_item] = '-'