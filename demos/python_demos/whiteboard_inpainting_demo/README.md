# Whiteboard Inpainting Demo

This demo focuses on a whiteboard text overlapped by a person. The demo shows
how to use the OpenVINO<sup>TM</sup> toolkit to detect and hide a person on a
video so that all text on a whiteboard is visible.

## How It Works

The demo expects one of the following models in the Intermediate Representation (IR) format:

* Instance segmentation model
* Semantic segmentation model

Use your own model or a pretrained model from the OpenVINO<sup>TM</sup> Open Model Zoo.
Find the list of models suitable for this demo in `models.lst`. Use the
[Model Downloader](../../../tools/downloader/README.md) to obtain the models.

As an input, the demo application takes:

* Path to a video file
* Index of a web camera

## Running

### Install Dependencies

To install required dependencies, open a terminal and run the following:

```bash
pip3 install -r requirements.txt
```

### Command-Line Arguments

Run the application with the `-h` option to see the following usage message:

```
usage: whiteboard_inpainting_demo.py [-h] -i I [-m_i M_INSTANCE_SEGMENTATION]
                                     [-m_s M_SEMANTIC_SEGMENTATION]
                                     [-t THRESHOLD]
                                     [--output_video OUTPUT_VIDEO] [--no_show]
                                     [-d DEVICE] [-l CPU_EXTENSION]
                                     [-u UTILIZATION_MONITORS]

Whiteboard inpainting demo

optional arguments:
  -h, --help            show this help message and exit
  -i I                  Input sources (index of camera or path to a video
                        file)
  -m_i M_INSTANCE_SEGMENTATION, --m_instance_segmentation M_INSTANCE_SEGMENTATION
                        Path to the instance segmentation model
  -m_s M_SEMANTIC_SEGMENTATION, --m_semantic_segmentation M_SEMANTIC_SEGMENTATION
                        Path to the semantic segmentation model
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold for person instance segmentation model
  --output_video OUTPUT_VIDEO
                        Optional. Path to output video
  --no_show             Optional. Don't show output
  -d DEVICE, --device DEVICE
                        Optional. Specify a target device to infer on. CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The demo will
                        look for a suitable plugin for the device specified
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with the kernels impl.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially
```

Example of a command:

```
python whiteboard_inpainting_demo.py \
    -i path/to/video.avi \
    -m_i path/to/instance-segmentation-security-0050.xml
```

## Demo output

The demo outputs original video with the processed one. Usage:

* Invert colors on the resulting frame by pressing the `i` key.
* Select a part of the frame to be shown in a separate window by using your left mouse button.
* Exit the demo by pressing `Esc`.
