# Multi-Channel Human Pose Estimation C++ Demo

This demo provides an inference pipeline for Multi-Channel Human Pose Estimation. The demo uses Human Pose Estimation network. You can use the following pre-trained model with the demos:
* `human-pose-estimation-0001`

For more information about the pre-trained models, refer to the [model documentation](../../../models/intel/index.md).

Other demo objectives are:

* Up to 16 cameras as inputs, via OpenCV*
* Visualization of detected faces from all channels on a single screen


## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. The Human Pose Estimation network is required.

> **NOTES**:
> * Running the demo requires using at least one web camera attached to your machine.
> * By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```
./multi_channel_human_pose_estimation_demo -h

multi_channel_human_pose_estimation_demo [OPTION]
Options:

    -h                           Print a usage message
    -i                           Required. A comma separated list of inputs to process. Each input must be a single image, a folder of images or anything that cv::VideoCapture can process.
    -loop                        Optional. Enable reading the inputs in a loop.
    -duplicate_num               Optional. Multiply the inputs by the given factor. For example, if only one input is provided, but -ni is set to 2, the demo uses half of images from the input as it was the first input and another half goes as the second input.
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
    -no_show                     Optional. Do not show processed video.
    -show_stats                  Optional. Enable statistics report
    -real_input_fps              Optional. Disable input frames caching, for maximum throughput pipeline
    -u                           Optional. List of monitors to show initially.
```

Running the application with an empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to run the demo with the pre-trained Human Pose Estimation model on FPGA with fallback on CPU with one camera, use the following command:
```sh
./multi_channel_human_pose_estimation_demo -m <path_to_model>/human-pose-estimation-0001.xml -d HETERO:FPGA,CPU -nc 1
```

To run the demo using two recorded video files, use the following command:
```sh
./multi_channel_human_pose_estimation_demo -m <path_to_model>/human-pose-estimation-0001.xml -d HETERO:FPGA,CPU -i /path/to/file1 /path/to/file2
```

Video files will be processed repeatedly.

You can also run the demo on web cameras and video files simultaneously by specifying both parameters: `-nc <number_of_cams> -i <video_file1> <video_file2>` with paths to video files separated by a space.
To run the demo with a single input source (a web camera or a video file), but several channels, specify an additional parameter: `-duplicate_num 3`. You will see four channels: one real and three duplicated. With several input sources, the `-duplicate_num` parameter will duplicate channels for each of them.

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
* [Model Downloader](../../../tools/downloader/README.md)
