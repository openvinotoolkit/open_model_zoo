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

class Evaluator(object):
    def __init__(self):
        '''Score Evaluation Variables'''
        self.video_eval_box = None
        self.eval_vars = None
        self.eval_cbs = None
        self.front_scores_df = None
        self.top_scores_df = None
        
    def initialize(self):
        self.first_put_take = False
        self.state = "Initial"
        self.use_tweezers_threshold = 50    # if tweezer and rider/weight distance more than tweezer treshold, consider use hand instead of use tweezer
        self.object_put = False # if battery is put on tray no matter left or right, return True
        self.weights_put = False    # if weight is put on tray no matter left or right, return True
        self.can_tidy = False # if battery and weight has been put on the tray(no matter left/right), then can tidy becomes true

        # scoring
        self.scoring = {
            "initial_score_rider":0,
            "initial_score_balance":0,
            "measuring_score_rider_tweezers":0,
            "measuring_score_balance":0,
            "measuring_score_object_left":0,
            "measuring_score_weights_right_tweezers":0,
            "measuring_score_weights_order":0,
            "end_score_tidy":0
        }

    def inference(self, top_det_results, front_det_results, top_seg_results, front_seg_results, frame_top, frame_front):
        """

        Args:
            top_det_results:
            front_det_results:
            top_seg_results:
            front_seg_results:

        Returns:
            Progress of the frame index
        """
        self.classify_state(top_seg_results,front_seg_results, frame_top, frame_front)
        if self.state == "Initial":
            self.evaluate_rider(front_seg_results,front_det_results)
            self.evaluate_scale_balance()

        elif self.state == "Measuring":
            if top_seg_results == "put_take":
                self.evaluate_object_left(front_seg_results,front_det_results)
                self.evaluate_weights_right_tweezers(front_seg_results,front_det_results)
                self.evaluate_weights_order()
                if not self.can_tidy:
                    self.can_tidy = self.evaluate_end_state()

            if top_seg_results == "adjust_rider":
                self.evaluate_rider_tweezers(front_seg_results,front_det_results)
            
        if self.can_tidy:
            self.evaluate_end_tidy()

        return self.state,self.scoring

    def classify_state(self,top_seg_results, front_seg_results, frame_top, frame_front):
        if top_seg_results == "put_take":
            self.first_put_take = True
        if self.first_put_take == True: 
            self.state = "Measuring"
        elif self.first_put_take == False:
            self.state = "Initial"

    def get_obj_coordinate(self,df):
        for i,row in df.iterrows():
            obj = row["obj"]
            x_min = row["x_min"]
            y_min = row["y_min"]
            x_max = row["x_max"]
            y_max = row["y_max"]
        return obj, x_min, y_min, x_max, y_max

    def evaluate_rider(self,front_seg_results,df):
        """
        Function:
            To evaluate whether rider is pushed to zero position

        Args:
            front_seg_results: action segmentation result of front view, eg: "noise_action" (type:string)
            df: front view object detection result a single frame (type: pandas dataframe)
        """
        # only evaluate rider zero if 2 roundscrew and 1 rider are found
        try:
            if df.obj.value_counts().roundscrew1 == 2 and df.obj.value_counts().rider == 1: 
                df_2screw = (df[df['obj']=="roundscrew1"])
                df_rider = (df[df['obj']=="rider"])

                roundscrew1_center_coordinate = (x0,y0) = ((df_2screw['x_max'].iloc[0] + df_2screw['x_min'].iloc[0])/2,(df_2screw['y_max'].iloc[0] + df_2screw['y_min'].iloc[0])/2)
                roundscrew2_center_coordinate = (x1,y1) = ((df_2screw['x_max'].iloc[1] + df_2screw['x_min'].iloc[1])/2,(df_2screw['y_max'].iloc[1] + df_2screw['y_min'].iloc[1])/2)
                rider_center_coordinate = (x3,y3) = ((df_rider['x_max'].iloc[0] + df_rider['x_min'].iloc[0])/2,(df_rider['y_max'].iloc[0] + df_rider['y_min'].iloc[0])/2)
                
                # if rider center position < 1/10 of length between 2 roundscrew, consider rider is pushed to zero position
                if rider_center_coordinate[0] < min(x0,x1)+abs(x0-x1)/10:
                    self.scoring["initial_score_rider"]=1
                else:
                    self.scoring["initial_score_rider"]=0
        except:
            pass

    def evaluate_rider_tweezers(self,front_seg_results,df):
        """
        Function:
            To evaluate whether rider is pushed using tweezers

        Args:
            front_seg_results: action segmentation result of front view, eg: "noise_action" (type:string)
            df: front view object detection result a single frame (type: pandas dataframe)

        """
        # only evaluate rider zero if 2 roundscrew and 1 rider are found
        try:
            if df.obj.value_counts().rider == 1 and df.obj.value_counts().tweezers == 1: 
                df_rider = (df[df['obj']=="rider"])
                df_tweezers = (df[df['obj']=="tweezers"])

                rider_min_coordinate = (x2,y2) = \
                    (df_rider['x_min'].iloc[0],df_rider['y_min'].iloc[0])
                tweezers_min_coordinate = (x3,y3) = \
                    (df_tweezers['x_min'].iloc[0],df_tweezers['y_min'].iloc[0])
                
                # if rider center position < 1/10 of length between 2 roundscrew, consider rider is pushed to zero position
                if abs(rider_min_coordinate-tweezers_min_coordinate)>self.use_tweezers_threshold:
                    self.scoring['measuring_score_rider_tweezers']=0
                else:
                    self.scoring['measuring_score_rider_tweezers']=1
        except:
            pass

    def evaluate_object_left(self,front_seg_results,df):

        if df.obj.value_counts().balance == 1 and df.obj.value_counts().battery == 1: 
            df_battery = (df[df['obj']=="battery"])
            df_balance = (df[df['obj']=="balance"])

            battery_center_coordinate = (x0,y0) = ((df_battery['x_max'].iloc[0] + df_battery['x_min'].iloc[0])/2,(df_battery['y_max'].iloc[0] + df_battery['y_min'].iloc[0])/2)
            (balance_x_min,balance_x_max,balance_y_min,balance_y_max) = (df_balance['x_min'].iloc[0],df_balance['x_max'].iloc[0],df_balance['y_min'].iloc[0],df_balance['y_max'].iloc[0] )

            if battery_center_coordinate[0]<(balance_x_min+(balance_x_max-balance_x_min)/2) and battery_center_coordinate[0]>balance_x_min:
                if battery_center_coordinate[1]>balance_y_min and battery_center_coordinate[1]<balance_y_max:
                    self.scoring['measuring_score_object_left'] = 1
                    self.put_object = 1
            if battery_center_coordinate[0]<balance_x_min and battery_center_coordinate[0]>balance_x_min:
                if battery_center_coordinate[1]>balance_y_min and battery_center_coordinate[1]<balance_y_max:
                    self.put_object = 1
            else:
                self.scoring['measuring_score_object_left'] = 0
    
    def evaluate_weights_right_tweezers(self,front_seg_results,df):
        if df.obj.value_counts().balance == 1 and "weights" in df.obj.values: 
            df_weights = (df[df['obj']=="weights"])
            df_balance = (df[df['obj']=="balance"])

            weights_center_coordinate = (x0,y0) = ((df_weights['x_max'].iloc[0] + df_weights['x_min'].iloc[0])/2,(df_weights['y_max'].iloc[0] + df_weights['y_min'].iloc[0])/2)
            (balance_x_min,balance_x_max,balance_y_min,balance_y_max) = (df_balance['x_min'].iloc[0],df_balance['x_max'].iloc[0],df_balance['y_min'].iloc[0],df_balance['y_max'].iloc[0] )

            if weights_center_coordinate[0]>(balance_x_min+(balance_x_max-balance_x_min)/2) and weights_center_coordinate[0]<balance_x_max:
                if weights_center_coordinate[1]>balance_y_min and weights_center_coordinate[1]<balance_y_max:
                    self.scoring['measuring_score_weights_right_tweezers'] = 1
                    self.weights_put = 1

            if weights_center_coordinate[0]>(balance_x_min+(balance_x_max-balance_x_min)/2) and weights_center_coordinate[0]<balance_x_max:
                if weights_center_coordinate[1]>balance_y_min and weights_center_coordinate[1]<balance_y_max:
                    self.weights_put = 1
            else:
                self.scoring['measuring_score_weights_right_tweezers'] = 0

    def evaluate_end_state(self):
        if self.object_put == 1:
            if self.weights_put ==1:
                self.can_tidy = True

    def evaluate_end_tidy(self,df):
        # define as tidied if no weights is detected(box closed),battery is on the table instead of the balance
        if df.obj.value_counts().weights == 0:
            if df.obj.value_counts().balance == 1 and df.obj.value_counts().battery == 1: 
                df_battery = (df[df['obj']=="battery"])
                df_balance = (df[df['obj']=="balance"])

                battery_center_coordinate = (x0,y0) = ((df_battery['x_max'].iloc[0] + df_battery['x_min'].iloc[0])/2,(df_battery['y_max'].iloc[0] + df_battery['y_min'].iloc[0])/2)
                (balance_x_min,balance_y_min,balance_x_max,balance_y_max) = (df_balance['x_min'].iloc[0],df_balance['y_min'].iloc[0],df_balance['x_max'].iloc[0],df_balance['y_max'].iloc[0] )
                if battery_center_coordinate[1]<balance_y_max:
                    self.scoring["end_score_tidy"] = 1


    def evaluate_scale_balance(self):
        pass

    def evaluate_weights_order(self):
        pass

    def evaluate_sleeve(self,df):
        # get balance coordinate
        filtered_df = df[df['obj'] == 'balance']
        balance, balance_x_min, balance_y_min, balance_x_max, balance_y_max = self.get_obj_coordinate(filtered_df)

        # get all support sleeve and pointer sleeve coordinate
        obj_list = ['support_sleeve','pointer_sleeve']
        filtered_df = df[df['obj'].isin(obj_list)]
        if len(filtered_df.index) == 0:     # if no sleeve detected, ie all taken away, get marks
            return True
        else:
            obj, x_min, y_min, x_max, y_max = self.get_obj_coordinate(filtered_df)

        try:
            # if all sleeves detected outside of balance, ie all sleeves has been taken down, get marks

            if x_min > balance_x_max or x_max < balance_x_min:
                return True

            if y_min > balance_y_max or y_max < balance_y_min:
                return True
            else:
                return False

        except:
            pass