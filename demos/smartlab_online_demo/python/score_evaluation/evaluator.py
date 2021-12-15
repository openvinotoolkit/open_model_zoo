# -- coding: utf-8 --
import cv2
import pandas as pd
import numpy as np

class Evaluator(object):
    def _init_(self):
        '''Score Evaluation Variables'''
        self.video_eval_box = None
        self.eval_vars = None
        self.eval_cbs = None
        self.front_scores_df = None
        self.top_scores_df = None
        self.buffer_rider = []    # buffer store coordinate of rider and tweezers to detect the move of rider, to evaluate the use of tweezers when adjust rider

    def initialize(self):
        self.first_put_take = False
        self.state = "Initial"
        self.use_tweezers_threshold = 200    # if tweezer and rider/weight distance more than tweezer treshold, consider use hand instead of use tweezer
        self.object_put = False # if battery is put on tray no matter left or right, return True
        self.weights_put = False    # if weight is put on tray no matter left or right, return True
        self.can_tidy = False # if battery and weight has been put on the tray(no matter left/right), then can tidy becomes true
        self.rider_move_threshold = 8 # if rider moves more than this value, check if tweezers or hand is used to move rider
        self.forgive_counter_rider = 0 # if forgive counter less than certain value, then forgive the not use of tweezers when adjust rider
        self.rider_zero=False
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

    def inference(self, top_det_results, front_det_results, top_seg_results, front_seg_results,frame_top,frame_front):
        """

        Args:
            top_det_results:
            front_det_results:
            top_seg_results:
            front_seg_results:

        Returns:
            Progress of the frame index
        """

        self.classify_state(top_seg_results,front_seg_results)
        if self.state == "Initial":
            self.evaluate_rider(front_seg_results,front_det_results)
            self.evaluate_scale_balance()

        elif self.state == "Measuring":
            if top_seg_results == "put_take":
                self.evaluate_object_left(front_seg_results,top_det_results)
                self.evaluate_weights_right_tweezers(front_seg_results,front_det_results)
                self.evaluate_weights_order()
                if not self.can_tidy:
                    self.evaluate_end_state()

            if top_seg_results == "adjust_rider":
                self.evaluate_rider_tweezers(top_seg_results,top_det_results)
            
        if self.can_tidy:
               self.evaluate_end_tidy(top_det_results,front_det_results,front_seg_results)
       
        return self.state,self.scoring

    def classify_state(self,top_seg_results, front_seg_results):
        if top_seg_results == "put_take":       #TODO:  filter input data for action so that only action persist  more than certain frame taken as true data
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
        if "roundscrew1" in df.obj.values and "rider" in df.obj.values:
            if df.obj.value_counts().roundscrew1 == 2 and df.obj.value_counts().rider == 1: 
                df_2screw = (df[df['obj']=="roundscrew1"])
                df_rider = (df[df['obj']=="rider"])

                roundscrew1_center_coordinate = (x0,y0) = ((df_2screw['x_max'].iloc[0] + df_2screw['x_min'].iloc[0])/2,(df_2screw['y_max'].iloc[0] + df_2screw['y_min'].iloc[0])/2)
                roundscrew2_center_coordinate = (x1,y1) = ((df_2screw['x_max'].iloc[1] + df_2screw['x_min'].iloc[1])/2,(df_2screw['y_max'].iloc[1] + df_2screw['y_min'].iloc[1])/2)
                rider_center_coordinate = (x3,y3) = ((df_rider['x_max'].iloc[0] + df_rider['x_min'].iloc[0])/2,(df_rider['y_max'].iloc[0] + df_rider['y_min'].iloc[0])/2)
                
                # if rider center position < 1/10 of length between 2 roundscrew, consider rider is pushed to zero position
                if rider_center_coordinate[0] < min(x0,x1)+abs(x0-x1)/10:
                    if self.state=="Initial":
                        self.scoring["initial_score_rider"]=1
                    else:
                        self.rider_zero=True
                else:
                    if self.state=="Initial":
                        self.scoring["initial_score_rider"]=0
                    else:
                        self.rider_zero=False
                    
    def evaluate_rider_tweezers(self,front_seg_results,df):
        """
        Function:
            To evaluate whether rider is pushed using tweezers

        Logic:
            if rider moves, tweezers coordinate should within certain pixels (defined in self.use_tweezers_threshold) from the rider coordinate

        """
        # only evaluate rider zero if 2 roundscrew and 1 rider are found
        if "rider" in df.obj.values and "tweezers" in df.obj.values:
            if df.obj.value_counts().rider == 1 and df.obj.value_counts().tweezers == 1: 
                df_rider = (df[df['obj']=="rider"])
                df_tweezers = (df[df['obj']=="tweezers"])

                rider_min_coordinate = (x2,y2) = (df_rider['x_min'].iloc[0],df_rider['y_min'].iloc[0])
                tweezers_min_coordinate = (x3,y3) = (df_tweezers['x_min'].iloc[0],df_tweezers['y_min'].iloc[0])
                
                self.buffer_rider.append(rider_min_coordinate[0])

                # if tweezers and rider apart more than use_tweezers_threshold pixels (based on x-coordinate only), consider not using tweezers
                if abs(rider_min_coordinate[0]-self.buffer_rider[0])>self.rider_move_threshold:
                    self.buffer_rider.pop(0)
                    if abs(rider_min_coordinate[0]-tweezers_min_coordinate[0])<self.use_tweezers_threshold:
                        self.scoring['measuring_score_rider_tweezers']=1

                    else:
                        self.forgive_counter_rider += 1
                        if self.forgive_counter_rider > 10:
                            self.scoring['measuring_score_rider_tweezers']=0


    def evaluate_object_left(self,front_seg_results,df):

        if "battery" in df.obj.values and "balance" in df.obj.values:
            if df.obj.value_counts().balance == 1 and df.obj.value_counts().battery == 1: 
                df_battery = (df[df['obj']=="battery"])
                df_balance = (df[df['obj']=="balance"])

                battery_center_coordinate = (x0,y0) = ((df_battery['x_max'].iloc[0] + df_battery['x_min'].iloc[0])/2,(df_battery['y_max'].iloc[0] + df_battery['y_min'].iloc[0])/2)
                (balance_x_min,balance_x_max,balance_y_min,balance_y_max) = (df_balance['x_min'].iloc[0],df_balance['x_max'].iloc[0],df_balance['y_min'].iloc[0],df_balance['y_max'].iloc[0] )


                # case1: object put at left
                if battery_center_coordinate[0]<(balance_x_min+(balance_x_max-balance_x_min)/2) and battery_center_coordinate[0]>balance_x_min and battery_center_coordinate[1]>balance_y_min and df_battery['y_min'].iloc[0]<balance_y_min+(balance_y_max-balance_y_min)/2:
                    self.scoring['measuring_score_object_left'] = 1
                    self.object_put = True
                # case2: object put at right
                elif battery_center_coordinate[0]<balance_x_min and battery_center_coordinate[0]>balance_x_min and battery_center_coordinate[1]>balance_y_min and df_battery['y_min'].iloc[0]<balance_y_min+(balance_y_max-balance_y_min)/2:
                    self.object_put = True
                # case3: object not put
                else:
                    self.scoring['measuring_score_object_left'] = 0
    

    def evaluate_weights_right_tweezers(self,front_seg_results,df):     #TODO:  determine whether tweezers are used or not
        if "weights" in df.obj.values and "balance" in df.obj.values and "tweezers" in df.obj.values:
            if df.obj.value_counts().balance == 1: 
                df_weights = (df[df['obj']=="weights"])
                df_balance = (df[df['obj']=="balance"])

                (balance_x_min,balance_x_max,balance_y_min,balance_y_max) = (df_balance['x_min'].iloc[0],df_balance['x_max'].iloc[0],df_balance['y_min'].iloc[0],df_balance['y_max'].iloc[0] )
                try:
                    df_weights = df_weights[df_weights['x_max']>balance_x_min+(balance_x_min+balance_x_max)/2] 
                    df_weights = df_weights[df_weights['x_min']<balance_x_max]
                    df_weights = df_weights[df_weights['y_max']>balance_y_min]
                    df_weights = df_weights[df_weights['y_min']<balance_y_max]
                   
                    self.weights_put = True
                    self.scoring['measuring_score_weights_right_tweezers'] = 1
                    
                except:
                    self.scoring['measuring_score_weights_right_tweezers'] = 0

    def evaluate_end_state(self):
        if self.object_put == True and self.weights_put == True:
            self.can_tidy = True


    def evaluate_end_tidy(self,top_det_results,front_det_results,front_seg_results):
        # define as tidied if no weights is detected(box closed),battery is on the table instead of the balance, and rider at zero position
        if "weights" in top_det_results.obj.values: #TODO: if no weight detected, and following condition, then only consider tidy
            pass
        elif "balance" in top_det_results.obj.values and "battery" in top_det_results.obj.values:
            if top_det_results.obj.value_counts().balance == 1 and top_det_results.obj.value_counts().battery == 1: 
                df_battery = (top_det_results[top_det_results['obj']=="battery"])
                df_balance = (top_det_results[top_det_results['obj']=="balance"])

                balance_coordinate = balance,balance_x_min,balance_y_min,balance_x_max,balance_y_max = self.get_obj_coordinate(df_balance)
                battery_coordinate = battery,battery_x_min,battery_y_min,battery_x_max,battery_y_max = self.get_obj_coordinate(df_battery)

                if battery_y_min>balance_y_min+(balance_y_max-balance_y_min)/2:
                    self.evaluate_rider(front_seg_results,front_det_results)
                    if self.rider_zero == True:
                        self.scoring["end_score_tidy"] = 1
                else:
                    self.scoring["end_score_tidy"] = 0


    def evaluate_scale_balance(self):
        pass

    def evaluate_weights_order(self):
        pass

    def evaluate_sleeve(self,df):
        
        # get balance coordinate
        filtered_df = df[df['obj'] == 'balance']
        balance, balance_x_min, balance_y_min, balance_x_max, balance_y_max = get_obj_coordinate(filtered_df)

        # get all support sleeve and pointer sleeve coordinate
        obj_list = ['support_sleeve','pointer_sleeve']
        filtered_df = df[df['obj'].isin(obj_list)]
        if len(filtered_df.index) == 0:     # if no sleeve detected, ie all taken away, get marks
            return True
        else:
            obj, x_min, y_min, x_max, y_max = get_obj_coordinate(filtered_df)

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
