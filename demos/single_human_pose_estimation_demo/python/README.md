# Single Human Pose Estimation Demo (top-down pipeline)

![example](./single_human_pose_estimation.gif)

This demo showcases top-down pipeline for human pose estimation on video or image. The task is to predict bboxes for every person on frame and then to predict a pose for every detected person. The pose may contain up to 17 keypoints: ears, eyes, nose, shoulders, elbows, wrists, hips, knees, and ankles.

## How It Works

On startup, the application reads command line parameters and loads detection person model and single human pose estimation model. Upon getting a frame from the OpenCV VideoCapture, the demo executes top-down pipeline for this frame and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/single_human_pose_estimation_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
python3 <omz_dir>/tools/downloader/downloader.py --list models.lst
```

An example of using the Model Converter:

```sh
python3 <omz_dir>/tools/downloader/converter.py --list models.lst
```

### Supported Models

* mobilenet-ssd
* pedestrian-and-vehicle-detector-adas-0001
* pedestrian-detection-adas-0002
* person-detection-retail-0013
* person-vehicle-bike-detection-crossroad-0078
* person-vehicle-bike-detection-crossroad-1016
* ssd300
* ssd512
* ssd_mobilenet_v1_coco
* ssd_mobilenet_v1_fpn_coco
* ssd_mobilenet_v2_coco
* ssdlite_mobilenet_v2
* single-human-pose-estimation-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: single_human_pose_estimation_demo.py [-h] -m_od MODEL_OD -m_hpe MODEL_HPE
                                            -i INPUT [--loop] [-o OUTPUT]
                                            [-limit OUTPUT_LIMIT] [-d DEVICE]
                                            [--person_label PERSON_LABEL]
                                            [--no_show]
                                            [-u UTILIZATION_MONITORS]

optional arguments:
  -h, --help            Show this help message and exit.
  -m_od MODEL_OD, --model_od MODEL_OD
                        Required. Path to model of object detector in .xml format.
  -m_hpe MODEL_HPE, --model_hpe MODEL_HPE
                        Required. Path to model of human pose estimator in .xml format.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a single image,
                        a folder of images, video file or camera id.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of the output file(s) to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target to infer on CPU or GPU.
  --person_label PERSON_LABEL
                        Optional. Label of class person for detector.
  --no_show             Optional. Do not display output.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

For example, to do inference on a CPU, run the following command:

```sh
python3 single_human_pose_estimation_demo.py -d CPU --model_od <path_to_model>/mobilenet-ssd.xml --model_hpe <path_to_model>/single-human-pose-estimation-0001.xml --input <path_to_video>/back-passengers.avi
```

When single image applied as an input, the demo will process and render it quickly, then exit. In this particular case, recommendation is to also apply `loop` option, which will enforce looping over processing the single image, so processed results will be continuously visualized on screen.
The demo allow saving of processed results to a Motion JPEG AVI file or separate JPEG or PNG files when `-o` option is used. To save processed results in AVI file, the name of output file with `avi` extension should be specified with `-o` option, for example: `-o output.avi`. To save processed results as an images, the template name of output image file with `jpg` or `png` extension should be specified with `-o` option, as shown on example: `-o output_%03d.jpg`. The actual file names will be constructed from template at runtime by replacing regular expression `%03d` with frame number, resulting in storing files with names like following: `output_000.jpg`, `output_001.jpg`, and so on.
Amount of data stored in output file or files could be limited with `limit` option, to avoid disk space overrun in case of continouus input stream, like camera. Default value is 1000, it can be changed by applying `-limit N` option, where N is number of frames to store.
In case folder of pictures is used as a demo input the recommendation is to store results as images too, storing to AVI file may not work if input images are differs in resolution.

>**NOTE**: Windows* systems may not have Motion JPEG codec installed by default. If this is the case, OpenCV FFMPEG backend could be downloaded by PowerShell script, located at OpenVINO install package at the path `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. This script should be run with Administrative privileges. Or, alternatively, storing results to images can be used.

## Demo Output

The demo uses OpenCV to display the resulting frame with estimated poses and reports performance in the following format: summary inference FPS (single human pose inference FPS / detector inference FPS).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
