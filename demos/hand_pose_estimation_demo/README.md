# Hand Pose Estimation C++ Demo

This demo showcases the work of hand pose estimation algorithm. The task is to predict a hand pose skeleton, which consists of keypoints and connections between them, for every person in an input image. Hand pose may contain up to 21 keypoints. You can use the following pre-trained model with the demo:

* `hand-pose-estimation`, which is a hand pose estimation network came from openpose.

For more information about the pre-trained model, refer to the link(https://github.com/CMU-Perceptual-Computing-Lab/openpose).

The input frame height is scaled to model height, frame width is scaled to preserve initial aspect ratio and padded to multiple of 8.

## How It Works

On the start-up, the application reads command line parameters and loads hand pose estimation model. Upon getting a frame from the OpenCV VideoCapture, the application executes hand pose estimation algorithm and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters][https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./hand_pose_estimation_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

hand_pose_estimation_demo [OPTION]
Options:

    -h                         Print a usage message.
    -i "<path>"                Required. Path to a image.
    -m "<path>"                Required. Path to the Hand Pose Estimation model (.xml) file.
    -d "<device>"              Optional. Specify the target device for Hand Pose Estimation (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -pc                        Optional. Enable per-layer performance report.
    -no_show                   Optional. Do not show processed video.
    -r                         Optional. Output inference results as raw values.

```

Running the application with an empty list of options yields an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to do inference on a CPU, run the following command:

```sh
./hand_pose_estimation_demo -i <path_to_image>/input_image.jpg -m <path_to_model>/hand-pose-estimation.xml -d CPU
```

## Demo Output

The demo uses OpenCV to display the resulting frame with estimated poses and text report of **FPS** - frames per second performance for the hand pose estimation demo.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
