# Pipeline for Online Video Analysis

The main file is `video_processor_serial.py`. In this ensemble version, given the pre-recorded videos, we aim to mimic the online process ( the results for frame k is calculated only by the historical frames)  .

###  Video Module

- Input

  - Receive the video path or online video address, include both the front view and top view

  - Parse the video stream and read them into the image array buffer and record the frame index simultaneously
  - Initialize the detection module (preloading the model and the corresponding network parameters) 
  - Initialize the action segmentation module  (preloading the model and the corresponding network parameters)
  - Initialize the score evaluation module  (preloading the model and the corresponding network parameters)
  - Load the video evaluation results

- Output (Threading?)

  - Pass the array of the current frame to the object detection module and call the object detection module
  - Pass the buffer of the current frame to the action segmentation module and call the action segmentation module 

### Object Detection Module 

Encapsulation into a class, support `self.initialize()` method and `self.inference()` method.

- Input
  - Image array of the current frame to be detected (two view)
- Output
  - Detection results

### Action Segmentation Module

- Input
  - Image array buffer of the two view (the sub module need to self judge and generate the temporal chunk for sliding window inference)
  - Index of the current latest frame
- Output
  - Segmentation results

### Score Evaluation Module

- Input
  - The detection results for the current frame and the temporal segmentation results for all the frames (two view).
- Output
  - Score results

# Next version 

> The main file is `video_processor_parallel.py`. In the next version,  we aim to process the task given the live videos of the two view.
>
> The threading techniques will be adopted for video reading and processing.....
