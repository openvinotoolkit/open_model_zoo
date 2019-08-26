# Multi Camera Multi Person Python* Demo

This demo demonstrates how to run Multi Camera Multi Person demo using OpenVINO<sup>TM</sup>

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5.2
* OpenVINO 2019 R2 with Python API

### Installation

To install required dependencies run

```bash
$ cat requirements.txt | xargs -n 1 -L 1 pip3 install
```

## Using

1. The demo expects the next models in the Intermediate Representation (IR) format:

   * Person detection model
   * Person re-identification model

It can be your own models or pre-trained model from OpenVINO Open Model Zoo.
In the `models.lst` are the list of appropriate models for this demo 
that can be obtained via `Model downloader`. 
Please see more information about `Model downloader` [here](https://github.com/opencv/open_model_zoo/blob/master/tools/downloader/README.md).

2. Inputs for the demo are videos or web-cameras.

3. Run the application with the `-h` option to see the following usage message:

```
usage: run.py [-h] [--videos VIDEOS [VIDEOS ...]]
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
4. Minimum command examples to run the demo:

```
# videos
python run.py \
    --videos path/to/video_1.avi path/to/video_2.avi \
    --pd_model path/to/person-detection-retail-0013.xml \
    --pr_model path/to/person-reidentification-retail-0076.xml 

# web-cameras
python run.py \
    --cam_ids 0 1 \
    --pd_model path/to/person-detection-retail-0013.xml \
    --pr_model path/to/person-reidentification-retail-0076.xml 
```

5. To save output video with the result please use the option  `--output_video`, to change configuration parameters please open the `config.py` file and edit it.
