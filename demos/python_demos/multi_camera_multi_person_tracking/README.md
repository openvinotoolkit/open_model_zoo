# Multi Camera Multi Person Python* Demo

This demo demonstrates how to run Multi Camera Multi Person demo using OpenVINO<sup>TM</sup>.

## How It Works

The demo expects the next models in the Intermediate Representation (IR) format:

   * Person detection model (or person instance segmentation model)
   * Person re-identification model

It can be your own models or pre-trained model from OpenVINO Open Model Zoo.
In the `models.lst` are the list of appropriate models for this demo
that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).

As input, the demo application takes:
* paths to several video files specified with a command line argument `--videos`
* indexes of web cameras specified with a command line argument `--cam_ids`

The demo workflow is the following:

1. The demo application reads tuples of frames from web cameras/videos one by one.
For each frame in tuple it runs person detector
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
usage: multi_camera_multi_person_tracking.py [-h] -i I [I ...]
                                             [--config CONFIG]
                                             [--detections DETECTIONS [DETECTIONS ...]]
                                             [-m M_DETECTOR]
                                             [--t_detector T_DETECTOR]
                                             [--m_segmentation M_SEGMENTATION]
                                             [--t_segmentation T_SEGMENTATION]
                                             --m_reid M_REID
                                             [--output_video OUTPUT_VIDEO]
                                             [--history_file HISTORY_FILE]
                                             [--no_show] [-d DEVICE]
                                             [-l CPU_EXTENSION]


Multi camera multi person tracking live demo script

optional arguments:
  -h, --help            show this help message and exit
  -i I [I ...]          Input sources (indexes of cameras or paths to video
                        files)
  --config CONFIG       Configuration file
  --detections DETECTIONS [DETECTIONS ...]
                        JSON files with detections
  -m M_DETECTOR, --m_detector M_DETECTOR
                        Path to the person detection model
  --t_detector T_DETECTOR
                        Threshold for the person detection model
  --m_segmentation M_SEGMENTATION
                        Path to the person instance segmentation model
  --t_segmentation T_SEGMENTATION
                        Threshold for person instance segmentation model
  --m_reid M_REID       Path to the person re-identification model
  --output_video OUTPUT_VIDEO
                        Optional. Path to output video
  --history_file HISTORY_FILE
                        Optional. Path to file in JSON format to save results
                        of the demo
  --no_show             Optional. Don't show output
  -d DEVICE, --device DEVICE
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with the kernels impl.

```
Minimum command examples to run the demo:

```
# videos
python multi_camera_multi_person_tracking.py \
    -i path/to/video_1.avi path/to/video_2.avi \
    --m_detector path/to/person-detection-retail-0013.xml \
    --m_reid path/to/person-reidentification-retail-0103.xml \
    --config config.py

# videos with instance segmentation model
python multi_camera_multi_person_tracking.py \
    -i path/to/video_1.avi path/to/video_2.avi \
    --m_segmentation path/to/instance-segmentation-security-0050.xml \
    --m_reid path/to/person-reidentification-retail-0107.xml \
    --config config.py

# web-cameras
python multi_camera_multi_person_tracking.py \
    -i 0 1 \
    --m_detector path/to/person-detection-retail-0013.xml \
    --m_reid path/to/person-reidentification-retail-0103.xml \
    --config config.py
```

The demo can use a JSON file with detections instead of a person detector.
The structure of this file should be as follows:
```json
[
    {
        'frame_id': 0
        'boxes': [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]  # N bounding boxes
        'scores': [score0, score1, ...]  # N scores
    },
    {
        'frame_id': 1
        'boxes': [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]
        'scores': [score0, score1, ...]
    }
]
```

## Demo Output

The demo displays bounding boxes of tracked objects and unique IDs of those objects.
To save output video with the result please use the option  `--output_video`,
to change configuration parameters please open the `config.py` file and edit it.

Visualization can be controlled using the following keys:
 * `space` - pause or next frame
 * `enter` - resume video
 * `esc` - exit

Also demo can dump resulting tracks to a json file. To specify the file use the
`--history_file` argument.
