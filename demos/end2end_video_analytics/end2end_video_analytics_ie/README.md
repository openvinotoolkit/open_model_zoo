# End to end video analytics end2end_video_analytics_ie Demo

(Ubuntu instructions..)

This tutorial demonstrate an end to end video analytics example.
The code includes few pipe stages.  
	1. Decode				..OpenCV decode h.264 video input
	2. Pre-processin		..OpenCV resize the image, format convert and prepare for inference
	3. Inference			..inference using the inference engine (SSD)
	4. Post Processing		..printing the labels and rendering the detection rectangular to video
	5. Encode				..encode using OpenCV

And you can merge normalization into first convolution as opposite to OpenCVDNN through Model Opimizer with --scale, --mean_values options.

Test contents are located in "samples/end2end_video_analytics/test_content" folder.

The stages to run the tutorial

1. "Model Downloader" --  Download the Deep Learning model using "Model Downloader"
2. "Model Optimizer"  --  Install prerequisites and run "Model Optimizer" to prepare the model for inference (and generate IR files)
3. Build inference engine samples
4. Run the tutorials



---------------------------------------------------------------------------------
1, "Model Downloader"

Download "MobileNet" model from the internet

$ cd /opt/intel/computer_vision_sdk/deployment_tools/model_downloader
$ sudo python3 downloader.py --name mobilenet-ssd


2. "Model Optimizer"

model optimizer prerequisites installation for caffe model

	$ sudo apt-get -y install python3-pip virtualenv cmake libpng12-dev libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libgstreamer0.10-dev libswscale-dev libavcodec-dev libavformat-dev python3-yaml
	$ cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/install_prerequisites
	$ sudo ./install_prerequisites.sh
	
	$ cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer
	$ source /opt/intel/computer_vision_sdk/bin/setupvars.sh
	$ source ./venv/bin/activate

We will run Model Optimizer twice, to generate FP32 and FP16 weights.

	// IR files, FP32 version, will be created in ~/Desktop/IR/FP32
	$ python3 ./mo.py --input_model ../model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel --scale 255.0 --mean_values [123.68,116.779,103.939] --output_dir ~/Desktop/IR/FP32 --data_type FP32

	// IR files, FP16 version, will be created in ~/Desktop/IR/FP16
	$ python3 ./mo.py --input_model ../model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel --scale 255.0 --mean_values [123.68,116.779,103.939] --output_dir ~/Desktop/IR/FP16 --data_type FP16

3. Build inference engine samples
   Open a new terminal.
   You can build from original inference_engine/samples folder with "sudo su" as well.

	$ cp -r /opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples ~/Desktop	
	$ cd ~/Desktop/samples	
	$ source /opt/intel/computer_vision_sdk/bin/setupvars.sh
	
	$ mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release; make
	
	$ sudo cp ./intel64/Release/lib/libcpu_extension.so /opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/.


## Running

4. Run the tutorials.   
   go to samples directory (ex: ~/Desktop/samples)

$ cd ~/Desktop/samples/build/intel64/Release

Present all the run flags and options:
$ ./end2end_video_analytics_ie -h

```sh

Run on CPU  (run SSD(Mobilenet) FP32 model, on a CPU )
$ ./end2end_video_analytics_ie -i ~/Desktop/samples/end2end_video_analytics/test_content/video/cars_768x768.h264 -d CPU -m ~/Desktop/IR/FP32/mobilenet-ssd.xml -l ~/Desktop/samples/end2end_video_analytics/test_content/IR/SSD/pascal_voc_classes.txt 

Run on GPU   (SSD-Mobilenet, FP16, 50 frames, batches of 5)
$ ./end2end_video_analytics_ie -i ~/Desktop/samples/end2end_video_analytics/test_content/video/cars_768x768.h264 -d GPU -m ~/Desktop/IR/FP16/mobilenet-ssd.xml -l ~/Desktop/samples/end2end_video_analytics/test_content/IR/SSD/pascal_voc_classes.txt -batch 5 -fr 50

Run on Movidius compute stick
$ ./end2end_video_analytics_ie -i ~/Desktop/samples/end2end_video_analytics/test_content/video/cars_768x768.h264 -d MYRIAD -m ~/Desktop/IR/FP16/mobilenet-ssd.xml -l ~/Desktop/samples/end2end_video_analytics/test_content/IR/SSD/pascal_voc_classes.txt 

```

### Outputs

The application outputs out.h264 (h264 video elementary stream with bounding box/ class label/ accuracy rate on the objects, you can play this with "$ mplayer out.h264").

### How it works

Upon the start-up of the demo application, it reads the command line parameters and loads a network and a video to the inference engine plugin. When the inference is done, the application will compose bounding boxes, class labels, and accuracy rates on the detected objects and encode it to h264 video elementary stream.

## See Also 
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
