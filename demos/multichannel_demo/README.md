# Multi-Channel Face Detection C++ Demo

This demo provides an inference pipeline for multi-channel face detection or human pose estimation. The demo uses Face Detection or Human Pose Estimation networks. You can use the following pre-trained models with the demo:
* `face-detection-retail-0004`
* `human-pose-estimation-0001`

For more information about the pre-trained models, refer to the [Open Model Zoo](https://github.com/opencv/open_model_zoo/tree/master/intel_models/index.md) repository on GitHub*.

Other demo objectives are:

* Up to 16 Cameras as inputs, via OpenCV\*
* Visualization of detected faces from all channels on a single screen


## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. The Face Detection network is required.

> **NOTES**:
> * Running the demo requires using at least one web camera attached to your machine.
> * By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Specify Input Shapes** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./multi-channel-face-detection-demo -h

multi-channel-face-detection-demo [OPTION]
Options:

    -h                           Print a usage message.
    -m "<path>"                  Required. Path to an .xml file with a trained face detection model.
      -l "<absolute_path>"       Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
          Or
      -c "<absolute_path>"       Required for GPU custom kernels. Absolute path to the xml file with the kernel descriptions.
    -d "<device>"                Optional. Specify the target device for Face Detection (CPU, GPU, FPGA, HDDL, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -nc                          Optional. Maximum number of processed camera inputs (web cameras)
    -bs                          Optional. Processing batch size, number of frames processed per infer request
    -n_ir                        Optional. Number of infer requests
    -n_iqs                       Optional. Frame queue size for input channels
    -fps_sp                      Optional. FPS measurement sampling period. Duration between timepoints, msec
    -n_sp                        Optional. Number of sampling periods
    -pc                          Optional. Enables per-layer performance report.
    -t                           Optional. Probability threshold for detections. Ignored for human pose estimation.
    -no_show                     Optional. Do not show processed video.
    -show_stats                  Optional. Enable statistics output
    -duplicate_num               Optional. Enable and specify a number of channel additionally copied from real sources
    -real_input_fps              Optional. Disable input frames caching, for maximum throughput pipeline
    -i                           Optional. Specify full path to input video files

```

Running the application with an empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to run the demo with the pre-trained face detection model on FPGA with fallback on CPU with one single camera, use the following command:
```sh
./multi-channel-face-detection-demo -m <path_to_model>/face-detection-retail-0004.xml
-l <demos_build_folder>/intel64/Release/lib/libcpu_extension.so -d HETERO:FPGA,CPU -nc 1
```

To run the demo using two recorded video files, use the following command:
```sh
./multi-channel-face-detection-demo -m <path_to_model>/face-detection-retail-0004.xml
-l <demos_build_folder>/intel64/Release/lib/libcpu_extension.so -d HETERO:FPGA,CPU -i /path/to/file1 /path/to/file2
```
Video files will be processed repeatedly.

You can also run the demo on web cameras and video files simultaneously by specifying both parameters: `-nc <number of cams> -i <video files sequentially, separated by space>`.
To run the demo with a single input source (a web camera or a video file) but several channels, specify an additional parameter: `-duplicate_num 3`. You will see four channels: one real and three duplicated. With several input sources, the `-duplicate_num` parameter will duplicate each of them.

The same applies for the `multi-channel-human-pose-estimation` demo except the model is `human-pose-estimation-0001.xml`.

## Input Video Sources

General parameter for input video source is `-i`. Use it to specify video files and web cameras (**USB cameras**) as input video source. You can add the parameter to the demo command line as follows:
```
    -i <file1> <file2>
```
`-nc <nc_value>` parameter simplifies usage of multiple web cameras. It connects web cameras with indexes from `0` to `nc_value-1`.
To see all available web cameras, run the `ls /dev/video*` command. You will get output similar to the following:
```sh
user@user-PC:~ $ ls /dev/video*
/dev/video0  /dev/video1  /dev/video2
```
You can use `-i` option to connect all the three web cameras:
```
    -i /dev/video0  /dev/video1  /dev/video2
```
Alternatively, you can just set `-nc 3`, which simplifies application usage.

If your cameras are connected to PC with indexes gap (for example, 0,1,3), use `-i` parameter.

IP-cameras through RSTP URI interface are not supported.


## Demo Output

The demo uses OpenCV to display the resulting bunch of frames with detections rendered as bounding boxes.
On the top of the screen, the demo reports throughput (in frames per second). If needed, it also reports more detailed statistics (use `-show_stats` option while running the demo to enable it).

## See Also
* [Using Open Model Zoo demos](https://github.com/opencv/open_model_zoo/tree/master/demos/README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader)
