# Instance Segmentation Python* Demo

This demo demonstrates how to run Instance Segmentation models from `Detectron` or `maskrcnn-benchmark` using OpenVINO&trade;.

> **NOTE**: Only batch size of 1 is supported.

## How It Works

The demo application expects an instance segmentation model in the Intermediate Representation (IR) format with the following constraints:

* Two inputs: `im_data` for input image and `im_info` for meta-information about the image (actual height, width and scale).
* At least four outputs containing:   
    * `boxes` with absolute bounding box coordinates of the input image   
    * `scores` with confidence scores for all bounding boxes  
    * `classes` with object class IDs for all bounding boxes   
    * `raw_masks` with fixed-size segmentation heat maps for all classes of all bounding boxes

As input, the demo application takes:
* a path to a single image file or a folder with image files specified with a command line argument `--images`
* a path to a video file or a numeric ID of a web-camera specified with a command line argument `--video`

The demo workflow is the following:

1. The demo application reads images/video frames one by one, resizes them to fit into the input image blob of the network (`im_data`) keeping the original image aspect ratio.
2. The `im_info` input blob passes resulting resolution and scale of a pre-processed image to the network to perform inference.
3. The demo visualizes the resulting instance segmentation masks. Certain command-line options affect the visualization:
    * If you specify `--show_boxes` and `--show_scores` arguments, bounding boxes and confidence scores are also shown.
    * If you specify video as a source of images with the `--video` option, the demo shows the same object instance with the same color throughout the whole video using simple tracking. It assumes more or less static scene with instances in two frames being a part of the same track if intersection over union of the masks is greater than the 0.5 threshold.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: main.py [-h] -m "<path>" --labels "<path>" [-v "<path>"] [-i "<path>"]
               [-d "<device>"] [-l "<absolute_path>"]
               [-pt "<num>"] [--show_scores] [--show_boxes] [-pc]

Options:
  -h, --help            Show this help message and exit.
  -m "<path>", --model "<path>"
                        Required. Path to an .xml file with a trained model.
  --labels "<path>"     Required. Path to a text file with class labels.
  -v "<path>", --video "<path>"
                        Required for video input. Path to a video file or
                        numeric camera ID.
  -i "<path>", --images "<path>"
                        Required for still images input. Path to an image or a
                        folder with images.
  -d "<device>", --device "<device>"
                        Optional. Specify the target device to infer on: CPU,
                        GPU, FPGA, HDDL or MYRIAD. The demo will look for a
                        suitable plugin for device specified (by default, it
                        is CPU).
  -l "<absolute_path>", --cpu_extension "<absolute_path>"
                        Required for CPU custom layers. Absolute path to a
                        shared library with the kernels implementation.
  -pt "<num>", --prob_threshold "<num>"
                        Optional. Probability threshold for detections
                        filtering.
  --show_scores         Optional. Show detection scores.
  --show_boxes          Optional. Show bounding boxes.
  -pc, --perf_counts    Optional. Report performance counters
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

To run the demo, please provide paths to the model in the IR format, to a file with class labels, and to an input video, image, or folder with images:
```bash
python3 instance_segmentation_demo/main.py -m <path_to_model>/maskrcnn_r50_fpn_2x.xml --label instance_segmentation_demo/coco_labels.txt --video 0
```

## Demo Output

The application uses OpenCV to display resulting instance segmentation masks and current inference performance.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
