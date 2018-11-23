# End to end video analytics end2end_video_analytics_opencv Demo

(Ubuntu instructions..)

This tutorial demonstrate an end to end video analytics example with OpenCV-DNN.
The code includes few pipe stages.  
	1. Decode				..OpenCV decode h.264 video input
	2. Pre-processin		..OpenCV resize the image, format convert and prepare for inference
	3. Inference			..inference using OpenCV-DNN (SSD model)
	4. Post Processing		..printing the labels and rendering the detection rectangular to video
	5. Encode				..encode using OpenCV

This tutorial demonstrates how to run image classification application, while utilizing OpenCV DNN for inferencing. 

Test contents are located in "samples/end2end_video_analytics/test_content" folder.

The stages to run the tutorial

1. "Model Downloader" --  Download the Deep Learning model using "Model Downloader"
2. Build the inference engine samples
3. Run the tutorials

---------------------------------------------------------------------------------
1, "Model Downloader"

$ cd /opt/intel/computer_vision_sdk/deployment_tools/model_downloader
$ sudo python3 downloader.py --name ssd300

2. Build inference engine samples
   Open a new terminal.
   You can build from original inference_engine/samples folder with "sudo su" as well.

	$ cp -r /opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples ~/Desktop	
	$ cd ~/Desktop/samples	
	$ source /opt/intel/computer_vision_sdk/bin/setupvars.sh
	
	$ mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release; make		

## Running

4. Run the tutorials.   
   go to samples directory (ex: ~/Desktop/samples)

```sh
$ cd ~/Desktop/samples/build/intel64/Release

$ ./end2end_video_analytics_opencv -h

[usage]
end2end_video_analytics_opencv [OPTION]
Options:

    -h                      
                            Print a usage message.
    -i "<path>"
                            Required. Path to input video file.
    -fr "<val>"             
                            Number of frames from stream to process.
    -m "<path>"            
                            Required. Path to Caffe deploy.prototxt file.
    -weights "<path>"           
                            Required. Path to Caffe weights in .caffemode file.
    -l "<path>"         
                            Required. Path to labels file.
    -s                     
                            Display less information on the screen.

```


You can run image classification on an image or a video using a trained network with multiple outputs on Intel&reg; Processors using the following command:

```sh

$ ./end2end_video_analytics_opencv -i ~/Desktop/samples/end2end_video_analytics/test_content/video/cars_768x768.h264 -m /opt/intel/deployment_tools/model_downloader/object_detection/common/ssd/300/caffe/ssd300.prototxt -weights /opt/intel/deployment_tools/model_downloader/object_detection/common/ssd/300/caffe/ssd300.caffemodel -l ~/Desktop/samples/end2end_video_analytics/test_content/IR/SSD/pascal_voc_classes.txt


```

### Outputs

The application outputs out.h264 (h264 video elementary stream with bounding box/ class label/ accuracy rate on the objects, you can play this with "$ mplayer out.h264").

### How it works

Upon the start-up of the demo application, it reads command line parameters and loads a network and an image or a video to the OpenCVDNN plugin. When the inference is done, the application will compose bounding boxes, class labels, and accuracy rates on the detected objects and encode it to h264 video elementary stream.

## See Also 
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
