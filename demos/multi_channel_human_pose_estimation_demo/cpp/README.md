# Multi-Channel Human Pose Estimation C++ Demo

This demo provides an inference pipeline for Multi-Channel Human Pose Estimation. The demo uses Human Pose Estimation network.

Other demo objectives are:

* Up to 16 cameras as inputs, via OpenCV*
* Visualization of detected faces from all channels on a single screen

## How It Works

On startup, the application reads command line parameters and loads the specified networks. The Human Pose Estimation network is required.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/multi_channel_human_pose_estimation_demo/cpp/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO IR format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models

* human-pose-estimation-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the demo with `-h` shows this help message:
```
    [-h]              Print a usage message
     -i               A comma separated list of inputs to process. Each input must be a single image, a folder of images or anything that cv::VideoCapture can process.
    [-loop]           Enable reading the inputs in a loop.
    [-duplicate_num]  Multiply the inputs by the given factor. For example, if only one input is provided, but -duplicate_num is set to 2, the demo will split real input across channels, by interleaving frames between channels.
     -m <path>        Path to an .xml file with a trained model.
    [-d <device>]     Specify a target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. Use "-d MULTI:<comma-separated_devices_list>" format to specify MULTI plugin. The application looks for a suitable plugin for the specified device.
    [-bs]             Batch size for processing (the number of frames processed per infer request)
    [-n_iqs]          Frame queue size for input channels
    [-fps_sp]         FPS measurement sampling period between timepoints in msec
    [-n_sp]           Number of sampling periods
    [-no_show]        Don't show output.
    [-show_stats]     Enable statistics report
    [-real_input_fps] Disable input frames caching, for maximum throughput pipeline
    [-u]              List of monitors to show initially.
```

To run the demo with the pre-trained Human Pose Estimation model on CPU with one camera, use the following command:

```sh
./multi_channel_human_pose_estimation_demo \
  -d CPU \
  -i 0 \
  -m <path_to_model>/human-pose-estimation-0001.xml
```

To run the demo using two recorded video files, use the following command:

```sh
./multi_channel_human_pose_estimation_demo \
  -d CPU \
  -i <path_to_file>/file1,<path_to_file>/file2 \
  -m <path_to_model>/human-pose-estimation-0001.xml
```

Video files will be processed simultaneously.

### Input Video Sources

General parameter for input source is `-i`. You can run the demo on web cameras and video files simultaneously by specifying: `-i <webcam_id0>,<webcam_id1>,<video_file1>,<video_file2>` with paths to webcams and video files separated by a comma. To run the demo with a single input source (a web camera or a video file), but several channels, specify an additional parameter, `duplicate_num`, for example: `-duplicate_num 4`. You will see four channels. With several input sources, the `-duplicate_num` parameter will duplicate each of them.

Below are some examples of demo input specification:

```sh
-i <file1>,<file2>
```

To see all available web cameras, run the `ls /dev/video*` command. You will get output similar to the following:

```sh
user@user-PC:~ $ ls /dev/video*
/dev/video0 /dev/video1 /dev/video2
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

The demo uses OpenCV to display the resulting frames with detections rendered as bounding boxes. The demo reports:

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).

You can also enable more detailed statistics in the output using the `-show_stats` option while running the demos.
You can use these metrics to measure application-level performance.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
