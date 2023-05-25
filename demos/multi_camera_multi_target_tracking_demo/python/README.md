# Multi Camera Multi Target Python\* Demo

This demo demonstrates how to run Multi Camera Multi Target (e.g. person or vehicle) demo using OpenVINO™.

## How It Works

The demo expects the following models in the Intermediate Representation (IR) format:

* object detection model or object instance segmentation model
* object re-identification model

As input, the demo application takes:

* paths to one or several video files
* numerical indexes of web cameras

The demo workflow is the following:

1. The demo application reads tuples of frames from web cameras/videos one by one.
For each frame in tuple it runs object detector
and then for each detected object it extracts embeddings using re-identification model.
2. All embeddings are passed to tracker which assigns an ID to each object.
3. The demo visualizes the resulting bounding boxes and unique object IDs assigned during tracking.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/multi_camera_multi_target_tracking_demo/python/models.lst` file.
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

* instance-segmentation-security-0002
* instance-segmentation-security-0091
* instance-segmentation-security-0228
* instance-segmentation-security-1039
* instance-segmentation-security-1040
* person-detection-retail-0013
* person-reidentification-retail-0277
* person-reidentification-retail-0286
* person-reidentification-retail-0287
* person-reidentification-retail-0288
* vehicle-reid-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

### Command Line Arguments

Run the application with the `-h` option to see the following usage message:

```
usage: multi_camera_multi_target_tracking_demo.py [-h] -i INPUT [INPUT ...]
                                                  [--loop] [--config CONFIG]
                                                  [--detections DETECTIONS]
                                                  [-m M_DETECTOR]
                                                  [--t_detector T_DETECTOR]
                                                  [--m_segmentation M_SEGMENTATION]
                                                  [--t_segmentation T_SEGMENTATION]
                                                  --m_reid M_REID
                                                  [--output_video OUTPUT_VIDEO]
                                                  [--history_file HISTORY_FILE]
                                                  [--save_detections SAVE_DETECTIONS]
                                                  [--no_show] [-d DEVICE]
                                                  [-u UTILIZATION_MONITORS]

Multi camera multi object tracking live demo script

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Required. Input sources (indexes of cameras or paths
                        to video files)
  --loop                Optional. Enable reading the input in a loop
  --config CONFIG       Configuration file
  --detections DETECTIONS
                        JSON file with bounding boxes
  -m M_DETECTOR, --m_detector M_DETECTOR
                        Path to the object detection model
  --t_detector T_DETECTOR
                        Threshold for the object detection model
  --m_segmentation M_SEGMENTATION
                        Path to the object instance segmentation model
  --t_segmentation T_SEGMENTATION
                        Threshold for object instance segmentation model
  --m_reid M_REID       Required. Path to the object re-identification model
  --output_video OUTPUT_VIDEO
                        Optional. Path to output video
  --history_file HISTORY_FILE
                        Optional. Path to file in JSON format to save results
                        of the demo
  --save_detections SAVE_DETECTIONS
                        Optional. Path to file in JSON format to save bounding
                        boxes
  --no_show             Optional. Don't show output
  -d DEVICE, --device DEVICE
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Minimum command examples to run the demo for person tracking (for vehicle tracking the commands are the same with appropriate vehicle detection/re-identification models):

```sh
# videos
python multi_camera_multi_target_tracking_demo.py \
    -i <path_to_video>/video_1.avi <path_to_video>/video_2.avi \
    --m_detector <path_to_model>/person-detection-retail-0013.xml \
    --m_reid <path_to_model>/person-reidentification-retail-0277.xml \
    --config configs/person.py

# videos with instance segmentation model
python multi_camera_multi_target_tracking_demo.py \
    -i <path_to_video>/video_1.avi <path_to_video>/video_2.avi \
    --m_segmentation <path_to_model>/instance-segmentation-security-0228.xml \
    --m_reid <path_to_model>/person-reidentification-retail-0277.xml \
    --config configs/person.py

# webcam
python multi_camera_multi_target_tracking_demo.py \
    -i 0 1 \
    --m_detector <path_to_model>/person-detection-retail-0013.xml \
    --m_reid <path_to_model>/person-reidentification-retail-0277.xml \
    --config configs/person.py
```

The demo can use a JSON file with detections instead of an object detector.
The structure of this file should be as follows:
```json
[
    [  # Source#0
        {
            "frame_id": 0,
            "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], ...],  # N bounding boxes
            "scores": [score0, score1, ...],  # N scores
        },
        {
            "frame_id": 1,
            "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], ...],
            "scores": [score0, score1, ...],
        },
        ...
    ],
    [  # Source#1
        {
            "frame_id": 0,
            "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], ...],  # N bounding boxes
            "scores": [score0, score1, ...],  # N scores
        },
        {
            "frame_id": 1,
            "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], ...],
            "scores": [score0, score1, ...],
        },
        ...
    ],
    ...
]
```

Such file with detections can be saved from the demo. Specify the argument `--save_detections` with path to an output file.

## Demo Output

The demo displays bounding boxes of tracked objects and unique IDs of those objects.
The demo reports

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).
You can use both of these metrics to measure application-level performance.

To save output video with the result please use the option  `--output_video`,
to change configuration parameters please open the `configs/person.py` (or `configs/vehicle.py` for vehicle tracking demo) file and edit it.

Visualization can be controlled using the following keys:

* `space` - pause or next frame
* `enter` - resume video
* `esc` - exit

Also demo can dump resulting tracks to a json file. To specify the file use the
`--history_file` argument.

## Quality Measuring

The demo provides tools for measure quality of the multi camera multi target tracker:

* Evaluation MOT metrics
* Visualize the demo results from a history file

For MOT metrics evaluation we use [py-motmetrics](https://github.com/cheind/py-motmetrics) module.
It is necessary to have ground truth annotation file for the evaluation. Supported format
of the ground truth annotation can be obtained via the annotation tool [CVAT](https://github.com/openvinotoolkit/cvat).
The annotation must includes the following labels and attributes:

```json
[
  {
    "name": "person",
    "id": 0,
    "attributes": [
      {
        "id": 0,
        "name": "id",
        "type": "text",
        "mutable": false,
        "values": [
          " "
        ]
      }
    ]
  }
]
```

To run evaluation MOT metrics use the following command:

```bash
python run_evaluate.py \
    --history_file <path_to_file>/file.json \
    --gt_files \
      <path_to_file>/source_0.xml \
      <path_to_file>/source_1.xml \
```

Number of ground truth files depends on the number of used video sources.

For the visualization of the demo results please use the next command:

```sh
python run_history_visualize.py \
    -i <path_to_video>/video_1.avi <path_to_video>/video_2.avi \
    --history_file <path_to_file>/file.json \
```

This a minimum arguments set for the script. To show all available arguments run the command with `-h` option:

```
usage: run_history_visualize.py [-h] [-i I [I ...]] --history_file
                                HISTORY_FILE [--output_video OUTPUT_VIDEO]
                                [--gt_files GT_FILES [GT_FILES ...]]
                                [--timeline TIMELINE] [--match_gt_ids]
                                [--merge_outputs]

Multi camera multi target tracking visualization demo script

optional arguments:
  -h, --help            show this help message and exit
  -i I [I ...]          Input videos
  --history_file HISTORY_FILE
                        File with tracker history
  --output_video OUTPUT_VIDEO
                        Output video file
  --gt_files GT_FILES [GT_FILES ...]
                        Files with ground truth annotation
  --timeline TIMELINE   Plot and save timeline
  --match_gt_ids        Match GT ids to ids from history
  --merge_outputs       Merge GT and history tracks into one frame
```

Ground truth files have the same format that was described in the MOT metrics evaluation part.

## Process Analysis

Two options are available during the demo execution:

1. Visualize distances between embeddings that are criterion for matching tracks.
2. Save and visualize embeddings (via `tensorboard`).

By default these options are disabled.
To enable the first one please set in configuration file for `analyzer` parameter
`enable` to `True`.

For the second one, install TensorBoard (for example, with `pip install tensorboard`).
Then, for `embeddings` specify parameter `save_path`
that is a directory where data related to embeddings will be saved
(if it is an empty string the option is disabled). There is paramater `use_images` in `embeddings`.
If it is `True` an image with object will be drawn for every embedding instead of point.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
