# Object Detection YOLO* V3 C++ Demo, Async API Performance Showcase

This demo showcases Object Detection with YOLO* V3 and Async API.

> **NOTE:** This topic describes usage of C++ implementation of the Object Detection YOLO* V3 Demo Async API . For the Python* implementation, refer to [Object Detection YOLO* V3 Python* Demo, Async API Performance Showcase](../python_demos/object_detection_demo_yolov3_async/README.md).

To learn more about Async API features, please refer to [Object Detection for SSD Demo, Async API Performance Showcase](../object_detection_demo_ssd_async/README.md).

Other demo objectives are:
* Video as input support via OpenCV*
* Visualization of the resulting bounding boxes and text labels (from the labels file, see `-labels` option) or class number (if no file is provided)
* OpenCV provides resulting bounding boxes, labels, and other information.
You can copy and paste this code without pulling Open Model Zoo demos helpers into your application
* Demonstration of the Async API in action. For this, the demo features two modes toggled by the **Tab** key:
    -  Old-style "Sync" way, where the frame captured with OpenCV executes back-to-back with the Detection
    -  Truly "Async" way, where the detection is performed on a current frame, while OpenCV captures the next frame

## How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```
./object_detection_demo_yolov3_async -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

object_detection_demo_yolov3_async [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path>"               Required. Path to a video file (specify "cam" to work with camera).
    -m "<path>"               Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Optional. Required for CPU custom layers. Absolute path to a shared library with the layers implementation.
          Or
      -c "<absolute_path>"    Optional. Required for GPU custom kernels. Absolute path to the .xml file with the kernels description.
    -d "<device>"             Optional. Specify a target device to infer on (the list of available devices is shown below). Default value is CPU. The demo will look for a suitable plugin for the specified device
    -labels "<path>"          Optional. Path to a file with labels mapping.
    -pc                       Optional. Enable per-layer performance report.
    -r                        Optional. Output inference results raw values showing.
    -t                        Optional. Probability threshold for detections.
    -iou_t                    Optional. Filtering intersection over union threshold for overlapping boxes.
    -auto_resize              Optional. Enable resizable input with support of ROI crop and auto resize.
    -no_show                  Optional. Do not show processed video.
    -u                        Optional. List of monitors to show initially.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

You can use the following command to do inference on GPU with a pre-trained object detection model:
```sh
./object_detection_demo_yolov3_async -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/yolo_v3.xml -d GPU
```

The only GUI knob is to use **Tab** to switch between the synchronized execution and the true Async mode.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode, the demo reports:
* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and to display the results.
* **Detection time**: inference time for the object detection network. It is reported in the Sync mode only.
* **Wallclock time**, which is combined application-level performance.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
