# Object Detection YOLO* V3 Python* Demo, Async API Performance Showcase

This demo showcases Object Detection with YOLO* V3 and Async API.

To learn more about Async API features, please refer to [Object Detection for SSD Demo, Async API Performance Showcase](../../object_detection_demo_ssd_async/README.md).

Other demo objectives are:
* Video as input support via OpenCV*
* Visualization of the resulting bounding boxes and text labels (from the `.labels` file) or class number (if no file is provided)
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
python3 object_detection_demo_yolov3_async.py -h
```
The command yields the following usage message:
```
usage: object_detection_demo_yolov3_async.py [-h] -m MODEL -i INPUT
                                       [-l CPU_EXTENSION] [-d DEVICE]
                                       [--labels LABELS] [-t PROB_THRESHOLD]
                                       [-iout IOU_THRESHOLD] [-ni NUMBER_ITER]
                                       [-pc] [-r]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to a image/video file. (Specify 'cam'
                        to work with camera)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU
  --labels LABELS       Optional. Labels mapping file
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections
                        filtering
  -iout IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD
                        Optional. Intersection over union threshold for
                        overlapping detections filtering
  -ni NUMBER_ITER, --number_iter NUMBER_ITER
                        Optional. Number of inference iterations
  -pc, --perf_counts    Optional. Report performance counters
  -r, --raw_output_message
                        Optional. Output inference results raw values showing
```

Running the application with the empty list of options yields the usage message given above and an error message.
You can use the following command to do inference on GPU with a pre-trained object detection model:
```
    python3 object_detection_demo_yolov3_async.py -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/yolo_v3.xml -d GPU
```

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

The only GUI knob is to use **Tab** to switch between the synchronized execution and the true Async mode.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode, the demo reports:
* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and to display the results.
* **Detection time**: inference time for the object detection network. It is reported in the Sync mode only.
* **Wallclock time**, which is combined application-level performance.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
* [YOLOv3 COCO labels](https://github.com/opencv/opencv/blob/master/samples/data/dnn/object_detection_classes_yolov3.txt), [VOC labels](https://github.com/opencv/opencv/blob/master/samples/data/dnn/object_detection_classes_pascal_voc.txt)
