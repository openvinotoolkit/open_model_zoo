# Multi-Channel Object Detection Yolov3 C++ Demo

This demo provides an multi-channel inference pipeline for YOLO v3 Object Detection network.

Other demo objectives are:

* Up to 16 cameras as inputs, via OpenCV*
* Visualization of detected objects from all channels on a single screen

## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. The Yolo v3 Object Detection network is required.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/multi_channel_object_detection_demo_yolov3/cpp/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin). You can also review OpenVINO [article](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html) to see how to convert the YOLO V3 and tiny YOLO V3 into IR model and execute this demo with converted IR model.

### Supported Models

* yolo-v3-tf
* yolo-v3-tiny-tf

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
multi_channel_object_detection_demo_yolov3 [OPTION]
Options:

    -h                           Print a usage message
    -i                           Required. A comma separated list of inputs to process. Each input must be a single image, a folder of images or anything that cv::VideoCapture can process.
    -loop                        Optional. Enable reading the inputs in a loop.
    -duplicate_num               Optional. Multiply the inputs by the given factor. For example, if only one input is provided, but -duplicate_num is set to 2, the demo will split real input across channels, by interleaving frames between channels.
    -m "<path>"                  Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"       Required for CPU custom layers. Absolute path to a shared library with the kernel implementations
          Or
      -c "<absolute_path>"       Required for GPU custom kernels. Absolute path to an .xml file with the kernel descriptions
    -d "<device>"                Optional. Specify the target device for a network (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo looks for a suitable plugin for a specified device.
    -bs                          Optional. Batch size for processing (the number of frames processed per infer request)
    -nireq                       Optional. Number of infer requests
    -n_iqs                       Optional. Frame queue size for input channels
    -fps_sp                      Optional. FPS measurement sampling period between timepoints in msec
    -n_sp                        Optional. Number of sampling periods
    -pc                          Optional. Enable per-layer performance report
    -t                           Optional. Probability threshold for detections
    -no_show                     Optional. Don't show output.
    -show_stats                  Optional. Enable statistics report
    -real_input_fps              Optional. Disable input frames caching, for maximum throughput pipeline
    -u                           Optional. List of monitors to show initially.
```

Running the application with an empty list of options yields the usage message given above and an error message.

For example, to run the demo on CPU, with one single camera, use the following command:

```sh
./multi_channel_object_detection_demo_yolov3 -m <path_to_model>/model.xml -d CPU -i 0
```

To run the demo on HDDL, using two recorded video files, use the following command:

```sh
./multi_channel_object_detection_demo_yolov3 -m <path_to_mdel>/model.xml -d HDDL -i <path_to_file>/file1,<path_to_file>/file2
```

Video files will be processed simultaneously.

To achieve 100% utilization of one Myriad X, the rule of thumb is to run 4 infer requests on each Myriad X. Option `-nireq 32` can be added to above command to use 100% of HDDL-R card. The 32 here is 8 (Myriad X on HDDL-R card) x 4 (infer requests), such as following command:

```sh
./multi_channel_object_detection_demo_yolov3 -m <path_to_model>/model.xml -d HDDL
-i <path_to_file>/file1,<path_to_file>/file2,<path_to_file>/file3,<path_to_file>/file4 -nireq 32
```

### Input Video Sources

General parameter for input source is `-i`. You can run the demo on web cameras and video files simultaneously by specifying: `-i <webcam_id0>,<webcam_id1>,<video_file1>,<video_file2>` with paths to webcams and video files separated by a comma. To run the demo with a single input source (a web camera or a video file), but several channels, specify an additional parameter, `duplicate_num`, for example: `-duplicate_num 4`. You will see four channels. With several input sources, the `-duplicate_num` parameter will duplicate each of them.

Below are some examples of demo input specification:

```sh
-i <file1>,<file2>
```

To see all available web cameras, run the `ls /dev/video*` command. You will get output similar to the following:

```sh
user@user-PC:~ $ ls /dev/video*
/dev/video0  /dev/video1  /dev/video2
```

You can use `-i` option to connect all the three web cameras:

```sh
-i 0,1,2
```

To connect to IP cameras, use RTSP URIs:

```sh
-i rtsp://camera_address_1/,rtsp://camera_address_2/
```

## Demo Output

The demo uses OpenCV to display the resulting frames with detections rendered as bounding boxes.
On the top of the screen, the demo reports throughput in frames per second. You can also enable more detailed statistics in the output using the `-show_stats` option while running the demos.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
