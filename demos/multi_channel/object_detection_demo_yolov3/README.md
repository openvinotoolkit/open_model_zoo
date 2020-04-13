# Multi-Channel Object Detection Yolov3 C++ Demo

This demo provides an inference pipeline for multi-channel yolo v3. The demo uses Yolo v3 Object Detection network. You can follow [this](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html) page convert the YOLO V3 and tiny YOLO V3 into IR model and execute this demo with converted IR model.

> **NOTES**:
> If you don't use [this](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html) page to convert the model, it may not work. 

Other demo objectives are:

* Up to 16 cameras as inputs, via OpenCV*
* Visualization of detected objects from all channels on a single screen


## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. The Yolo v3 Object Detection network is required.

> **NOTES**:
> * By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```
cd <samples_build_folder>/intel64/Release
./multi_channel_object_detection_demo_yolov3 -h

multi_channel_object_detection_demo_yolov3 [OPTION]
Options:

    -h                           Print a usage message.
    -m "<path>"                  Required. Path to an .xml file with a trained yolo v3 or tiny yolo v3 model.
      -l "<absolute_path>"       Required for MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"       Required for clDNN (GPU)-targeted custom kernels. Absolute path to the xml file with the kernels desc.
    -d "<device>"                Optional. Specify the target device for Face Detection (CPU, GPU, FPGA, HDDL or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -nc                          Optional. Maximum number of processed camera inputs (web cams)
    -bs                          Optional. Batch size for processing (the number of frames processed per infer request)
    -nireq                       Optional. Number of infer requests
    -n_iqs                       Optional. Frame queue size for input channels
    -fps_sp                      Optional. FPS measurement sampling period. Duration between timepoints, msec
    -n_sp                        Optional. Number of sampling periods
    -pc                          Optional. Enables per-layer performance report.
    -t                           Optional. Probability threshold for detections.
    -no_show                     Optional. No show processed video.
    -show_stats                  Optional. Enable statistics report
    -duplicate_num               Optional. Enable and specify number of channel additionally copied from real sources
    -real_input_fps              Optional. Disable input frames caching, for maximum throughput pipeline
    -i                           Optional. Specify full path to input video files
    -loop_video                  Optional. Enable playing video on a loop.
    -u                           Optional. List of monitors to show initially.
```

To run the demo, you can use public pre-train model and follow [this](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html) page for instruction of how to convert it to IR model. 

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to run the demo on FPGA with fallback on CPU, with one single camera, use the following command:
```sh
./multi_channel_object_detection_demo_yolov3 -m $PATH_OF_YOLO_V3_MODEL -d HETERO:FPGA,CPU -nc 1
```

To run the demo using two recorded video files, use the following command:
```sh
./multi_channel_object_detection_demo_yolov3 -m $PATH_OF_YOLO_V3_MODEL -d HDDL -i /path/to/file1 /path/to/file2
```
Video files will be processed repeatedly.

To achieve 100% utilization of one Myriad X, the thumb rule is to run 4 infer requests on each Myriad X. Option `-nireq 32` can be added to above command to use 100% of HDDL-R card. The 32 here is 8 (Myriad X on HDDL-R card) x 4 (infer requests), such as following command:

```sh
./multi_channel_object_detection_demo_yolov3 -m $PATH_OF_YOLO_V3_MODEL -d HDDL 
-i /path/to/file1 /path/to/file2 /path/to/file3 /path/to/file4 -nireq 32
```

You can also run the demo on web cameras and video files simultaneously by specifying both parameters: `-nc <number of cams> -i <video files sequentially, separated by space>`.
To run the demo with a single input source(a web camera or a video file), but several channels, specify an additional parameter: `-duplicate_num 3`. You will see four channels: one real and three duplicated. With several input sources, the `-duplicate_num` parameter will duplicate each of them.

## Demo Output

The demo uses OpenCV to display the resulting frames with detections rendered as bounding boxes.
On the top of the screen, the demo reports throughput in frames per second. You can also enable more detailed statistics in the output using the `-show_stats` option while running the demos.


## Input Video Sources

General parameter for input video source is `-i`. Use it to specify video files or web cameras as input video sources. You can add the parameter to a sample command line as follows:
```
-i <file1> <file2>
```

`-nc <nc_value>` parameter simplifies usage of multiple web cameras. It connects web cameras with indexes from `0` to `nc_value-1`.

To see all available web cameras, run the `ls /dev/video*` command. You will get output similar to the following:

```
user@user-PC:~ $ ls /dev/video*
/dev/video0  /dev/video1  /dev/video2
```

You can use `-i` option to connect all the three web cameras:

```
-i /dev/video0  /dev/video1  /dev/video2
```

Alternatively, you can just set `-nc 3`, which simplifies application usage.

If your cameras are connected to PC with indexes gap (for example, `0,1,3`), use the `-i` parameter.

To connect to IP cameras, use RTSP URIs:
```
-i rtsp://camera_address_1/ rtsp://camera_address_2/
```

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
