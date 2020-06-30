# Object Detection RetinaFace Demo

This demo showcases Face Detection with RetinaFace. The task is to identify faces as axis-aligned boxes
and their keypoints (facial landmarks) in an image.

## How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

## Running

Running the application with the <code>-h</code> option yields the following usage message:

```
usage: object_detection_demo_retinaface.py [-h] -m MODEL
                                           [-i INPUT [INPUT ...]] [-d DEVICE]
                                           [-pt_f FACE_PROB_THRESHOLD]
                                           [-pt_m MASK_PROB_THRESHOLD]
                                           [--no_show]
                                           [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        path to video or image/images
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The demo will
                        look for a suitable plugin for device specified.
                        Default value is CPU
  -pt_f FACE_PROB_THRESHOLD, --face_prob_threshold FACE_PROB_THRESHOLD
                        Optional. Probability threshold for face detections
                        filtering
  -pt_m MASK_PROB_THRESHOLD, --mask_prob_threshold MASK_PROB_THRESHOLD
                        Optional. Probability threshold for mask detections
                        filtering
  --no_show             Optional. Don't show output
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

## Demo Output

The demo uses OpenCV to display the resulting frame with detections and reports performance in the following format: summary inference FPS.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
