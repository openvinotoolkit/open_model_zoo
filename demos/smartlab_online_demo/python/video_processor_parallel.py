# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import time
from collections import deque
import threading
from object_detection.detector import Detector
from temporal_segmentation.segmentor import Segmentor
from score_evaluation.evaluator import Evaluator


class Video_Parser(object):

    def __init__(self):
        self.buffer = deque(maxlen=1000)  # Array buffer TODO How to delete? manually or designing rules?
        self.frame_counter = 0  # Frame index counter
        self.playing = True  # Control button for video processing

    def capture(self, video_path, view):
        """
            Read the video stream and store the image in the buffer && record the frame index
        Args:
            video_path:
            view:
        """
        self.cap = cv2.VideoCapture(video_path)

        while self.cap.isOpened() and self.playing:
            ret, frame = self.cap.read()  # frame:480 x 640 x 3
            if ret:
                self.frame_counter += 1
                self.buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                self.playing = False
                print("Finished for the %s view!" % view)
                break


class Application(object):
    """
        Video process for the live camera (demo only )
    """
    def __init__(self, ):
        """
            Initialize Variables
        """
        ''' Progress Variables'''
        self.det_process_counter = 0  # Number of frames processed by the object detection module
        self.seg_process_counter = 0  # Number of frames processed by the temporal segmentation module
        self.eval_process_counter = 0  # Number of frames processed by the score evaluation module

        ''' Object Detection Variables'''
        self.detector = Detector()
        self.detector.initialize()  # Initialize the session and load the model parameters

        '''Video Segmentation Variables'''
        self.segmentor = Segmentor()
        self.segmentor.initialize()  # Initialize the session and load the model parameters

        '''Score Evaluation Variables'''
        self.evaluator = Evaluator()
        self.evaluator.initialize()

    def video_parser(self, cap_top, cap_front):
        """
        Process the video.
        """
        while self.eval_process_counter <= min(cap_top.frame_counter, cap_front.frame_counter):
            ''' The object detection module need to self judge and generate detection results in the given folder '''
            self.detector.inference(self.buffer_top, self.buffer_front)

            ''' The temporal segmentation module need to self judge and generate segmentation results in the given folder '''
            self.segmentor.inference(self.buffer_top, self.buffer_front)

            ''' The score evaluation module need to self merge the results of the two modules in the given folder 
                and generate the score results '''
            self.evaluator.inference(self.detector, self.segmentor)

        print("Finished!")

    def get_video_fps(self, cap):
        return cap.get(cv2.CAP_PROP_FPS)

    def get_video_total_frames(self, cap):
        return cap.get(cv2.CAP_PROP_FRAME_COUNT)


if __name__ == "__main__":
    application = Application()

    cap_top = Video_Parser()
    cap_front = Video_Parser()
    t1 = threading.Thread(target=cap_top.capture,
                          args=("/home/pingguo/wb/scale_balance_evaluation_online/data/video_9_top.avi", "top"))
    t2 = threading.Thread(target=cap_front.capture,
                          args=("/home/pingguo/wb/scale_balance_evaluation_online/data/video_9_front.avi", "front"))
    t1.start()
    t2.start()
    # TODO sync between two caps?
    application.video_parser(cap_top=cap_top, cap_front=cap_front)
