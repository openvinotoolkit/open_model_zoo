# Human Pose Estimation C++ Demo

![example](../human_pose_estimation.gif)

This demo showcases the work of multi-person 2D pose estimation algorithm. The task is to predict a pose: body skeleton, which consists of keypoints and connections between them, for every person in an input video. The pose may contain up to 18 keypoints: *ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees*, and *ankles*. Some of potential use cases of the algorithm are action recognition and behavior understanding.

Other demo objectives are:

* Video/Camera as inputs, via OpenCV*
* Visualization of all estimated poses

## How It Works

On the start-up, the application reads command line parameters and loads human pose estimation model. Upon getting a frame from the OpenCV VideoCapture, input frame height is scaled to model height, frame width is scaled to preserve initial aspect ratio and padded to multiple of 8, then application executes human pose estimation algorithm and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
Pre-trained models, supported by demo listed in [models.lst](./models.lst) file, located at each demo folder.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported models

* architecture_type = openpose
    - human-pose-estimation-0001
* architecture_type = ae
    - human-pose-estimation-0005
    - human-pose-estimation-0006
    - human-pose-estimation-0007
* architecture_type = higherhrnet
    - higher-hrnet-w32-human-pose-estimation

> **NOTE**: Refer to tables for [Intel](../../../models/intel/device_support.md) and [public](../../../models/public/device_support.md) models which summarize models support at different devices to select target inference device.

## Running

Running the application with the `-h` option yields the following usage message:

```
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

human_pose_estimation_demo [OPTION]
Options:

    -h                        Print a usage message.
    -at "<type>"              Required. Type of the network, either 'ae' for Associative Embedding, 'higherhrnet' for HigherHRNet models based on ae or 'openpose' for OpenPose.
    -i                        Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -m "<path>"               Required. Path to an .xml file with a trained model.
    -o "<path>"               Optional. Name of output to save.
    -limit "<num>"            Optional. Number of frames to store in output. If 0 is set, all frames are stored.
    -tsize                    Optional. Target input size.
      -l "<absolute_path>"    Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
          Or
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with the kernel descriptions.
    -d "<device>"             Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -pc                       Optional. Enables per-layer performance report.
    -t                        Optional. Probability threshold for poses filtering.
    -nireq "<integer>"        Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.
    -nthreads "<integer>"     Optional. Number of threads.
    -nstreams                 Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)
    -loop                     Optional. Enable reading the input in a loop.
    -no_show                  Optional. Don't show output.
    -output_resolution        Optional. Specify the maximum output window resolution in (width x height) format. Example: 1280x720. Input frame size used by default.
    -u                        Optional. List of monitors to show initially.
```

Running the application with an empty list of options yields an error message.

For example, to do inference on a CPU, run the following command:

```sh
./human_pose_estimation_demo -i <path_to_video>/input_video.mp4 -m <path_to_model>/human-pose-estimation-0001.xml -d CPU -at openpose
```

## Demo Output

The demo uses OpenCV to display the resulting frame with estimated poses and text report of **FPS** - frames per second performance for the human pose estimation demo.

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
