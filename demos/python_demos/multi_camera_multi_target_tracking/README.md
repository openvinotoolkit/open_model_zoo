# Multi Camera Multi Target Python* Demo

This demo demonstrates how to run Multi Camera Multi Target (e.g. person or vehicle) demo using OpenVINO<sup>TM</sup>.

## How It Works

The demo expects the next models in the Intermediate Representation (IR) format:

   * Object detection model (or object instance segmentation model)
   * Object re-identification model

It can be your own models or pre-trained model from OpenVINO Open Model Zoo.
In the `models.lst` are the list of appropriate models for this demo
that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).

As input, the demo application takes:
* paths to several video files specified with a command line argument `--videos`
* indexes of web cameras specified with a command line argument `--cam_ids`

The demo workflow is the following:

1. The demo application reads tuples of frames from web cameras/videos one by one.
For each frame in tuple it runs object detector
and then for each detected object it extracts embeddings using re-identification model.
2. All embeddings are passed to tracker which assigns an ID to each object.
3. The demo visualizes the resulting bounding boxes and unique object IDs assigned during tracking.

## Running

### Installation of dependencies

To install required dependencies run

```bash
pip3 install -r requirements.txt
```

### Command line arguments

Run the application with the `-h` option to see the following usage message:

```
usage: multi_camera_multi_target_tracking.py [-h] -i I [I ...]
                                             [--config CONFIG]
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
                                             [-l CPU_EXTENSION]
                                             [-u UTILIZATION_MONITORS]

Multi camera multi target tracking live demo script

optional arguments:
  -h, --help            show this help message and exit
  -i I [I ...]          Input sources (indexes of cameras or paths to video
                        files)
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
  --m_reid M_REID       Path to the object re-identification model
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
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with the kernels impl.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```
Minimum command examples to run the demo for person tracking (for vehicle tracking the commands are the same with appropriate vehicle detection/re-identification models):

```
# videos
python multi_camera_multi_target_tracking.py \
    -i path/to/video_1.avi path/to/video_2.avi \
    --m_detector path/to/person-detection-retail-0013.xml \
    --m_reid path/to/person-reidentification-retail-0103.xml \
    --config configs/person.py

# videos with instance segmentation model
python multi_camera_multi_person_tracking.py \
    -i path/to/video_1.avi path/to/video_2.avi \
    --m_segmentation path/to/instance-segmentation-security-0050.xml \
    --m_reid path/to/person-reidentification-retail-0107.xml \
    --config configs/person.py

# web-cameras
python multi_camera_multi_person_tracking.py \
    -i 0 1 \
    --m_detector path/to/person-detection-retail-0013.xml \
    --m_reid path/to/person-reidentification-retail-0103.xml \
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
Such file with detections can be saved from the demo. Specify the argument
`--save_detections` with path to an output file.

## Demo Output

The demo displays bounding boxes of tracked objects and unique IDs of those objects.
To save output video with the result please use the option  `--output_video`,
to change configuration parameters please open the `configs/person.py` (or `configs/vehicle.py` for vehicle tracking demo) file and edit it.

Visualization can be controlled using the following keys:
 * `space` - pause or next frame
 * `enter` - resume video
 * `esc` - exit

Also demo can dump resulting tracks to a json file. To specify the file use the
`--history_file` argument.

## Quality measuring

The demo provides tools for measure quality of the multi camera multi target tracker:
* Evaluation MOT metrics
* Visualize the demo results from a history file

For MOT metrics evaluation we use [py-motmetrics](https://github.com/cheind/py-motmetrics) module.
It is necessary to have ground truth annotation file for the evaluation. Supported format
of the ground truth annotation can be obtained via the annotation tool [CVAT](https://github.com/opencv/cvat).
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
    --history_file path/to/history/file.json \
    --gt_files \
      path/to/ground/truth/annotation/for/source_0.xml \
      path/to/ground/truth/annotation/for/source_1.xml \
```
Number of ground truth files depends on the number of used video sources.

For the visualization of the demo results please use the next command:
```
python run_history_visualize.py \
    -i path/to/video_1.avi path/to/video_2.avi \
    --history_file path/to/history/file.json \
```
This a minimum arguments set for the script. To show all available arguments
run the command:
```
python3 run_history_visualize.py -h
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
Ground truth files have the same format was described in the MOT metrics evaluation part.

## Process analysis

During the demo execution are available two options for analysis the process:
1. Visualize distances between embeddings that are criterion for matching tracks.
2. Save and visualize embeddings (via `tensorboard`).

By default these options are disabled.
To enable the first one please set in configuration file for `analyzer` parameter
`enable` to `True`, for the second one for `embeddings` specify parameter `path`
that is a directory where data related to embeddings will be saved
(if it is an empty string the option is disabled). In `embeddings` is a parameter
`use_images`. If it is `True` for every embedding will be drawn an image with an object
instead a point.
