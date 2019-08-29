# Multi Camera Multi Person Python* Demo

This demo demonstrates how to run Multi Camera Multi Person demo using OpenVINO<sup>TM</sup>.

## How It Works

The demo expects the next models in the Intermediate Representation (IR) format:

   * Person detection model
   * Person re-identification model

It can be your own models or pre-trained model from OpenVINO Open Model Zoo.
In the `models.lst` are the list of appropriate models for this demo
that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).

As input, the demo application takes:
* paths to several video files specified with a command line argument `--videos`
* indexes of web cameras specified with a command line argument `--cam_ids`

The demo workflow is the following:

1. The demo application reads tuples of frames from web cameras/videos one by one. For each frame in tuple it runs person detector
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
usage: multi_camera_multi_person_tracking.py [-h] [--videos VIDEOS [VIDEOS ...]]
              [--cam_ids CAM_IDS [CAM_IDS ...]] --pd_model PD_MODEL
              [--pd_thresh PD_THRESH] [--pr_model PR_MODEL]
              [--pr_thresh PR_THRESH] [--output_video OUTPUT_VIDEO]
              [--config CONFIG] [--history_file HISTORY_FILE]
              [--device DEVICE] [-l CPU_EXTENSION]

Multi camera multi person tracking live demo script

optional arguments:
  -h, --help            show this help message and exit
  --videos VIDEOS [VIDEOS ...]
                        Input videos
  --cam_ids CAM_IDS [CAM_IDS ...]
                        Indexes of input cameras
  --pd_model PD_MODEL
  --pd_thresh PD_THRESH
                        Threshold for person detection model
  --pr_model PR_MODEL
  --pr_thresh PR_THRESH
                        Threshold for person re-identification model
  --output_video OUTPUT_VIDEO
  --config CONFIG
  --history_file HISTORY_FILE
  --device DEVICE
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with the kernels impl.
```
Minimum command examples to run the demo:

```
# videos
python multi_camera_multi_person_tracking.py \
    --videos path/to/video_1.avi path/to/video_2.avi \
    --pd_model path/to/person-detection-retail-0013.xml \
    --pr_model path/to/person-reidentification-retail-0076.xml \
    --config config.py

# web-cameras
python multi_camera_multi_person_tracking.py \
    --cam_ids 0 1 \
    --pd_model path/to/person-detection-retail-0013.xml \
    --pr_model path/to/person-reidentification-retail-0076.xml \
    --config config.py
```

## Demo Output

The demo displays bounding boxes of tracked objects and unique IDs of those objects.
To save output video with the result please use the option  `--output_video`, to change configuration parameters please open the `config.py` file and edit it.

Also demo can dump resulting tracks to a json file. To specify the file use the `--history_file` argument.
