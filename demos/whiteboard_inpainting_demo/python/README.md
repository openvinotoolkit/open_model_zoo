# Whiteboard Inpainting Python\* Demo

![example](./whiteboard_inpainting.gif)

This demo focuses on a whiteboard text overlapped by a person. The demo shows
how to use the OpenVINO™ toolkit to detect and hide a person on a
video so that all text on a whiteboard is visible.

## How It Works

The demo expects one of the following models in the Intermediate Representation (IR) format:

* Instance segmentation model
* Semantic segmentation model

As an input, the demo application takes:

* Path to a video file
* Index of a web camera

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/whiteboard_inpainting_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported Models

* instance-segmentation-security-0002
* instance-segmentation-security-0091
* instance-segmentation-security-0228
* instance-segmentation-security-1039
* instance-segmentation-security-1040
* semantic-segmentation-adas-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

### Install Dependencies

To install required dependencies, open a terminal and run the following:

```bash
pip3 install -r requirements.txt
```

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: whiteboard_inpainting_demo.py [-h] -i INPUT [--loop] [-o OUTPUT]
                                     [-limit OUTPUT_LIMIT]
                                     -m_i M_INSTANCE_SEGMENTATION
                                     -m_s M_SEMANTIC_SEGMENTATION
                                     [-t THRESHOLD] [--no_show]
                                     [-d DEVICE] [-l CPU_EXTENSION]
                                     [-u UTILIZATION_MONITORS]

Whiteboard inpainting demo

optional arguments:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. Path to a video file or a device node of a
                        webcam.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  -m_i M_INSTANCE_SEGMENTATION, --m_instance_segmentation M_INSTANCE_SEGMENTATION
                        Required. Path to the instance segmentation model.
  -m_s M_SEMANTIC_SEGMENTATION, --m_semantic_segmentation M_SEMANTIC_SEGMENTATION
                        Required. Path to the semantic segmentation model.
  -t THRESHOLD, --threshold THRESHOLD
                        Optional. Threshold for person instance segmentation model.
  --no_show             Optional. Don't show output.
  -d DEVICE, --device DEVICE
                        Optional. Specify a target device to infer on. CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The demo will
                        look for a suitable plugin for the device specified.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers. Absolute path to a
                        shared library with the kernels impl.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Example of a command:

```sh
python3 whiteboard_inpainting_demo.py \
    -i <path_to_video>/video.avi \
    -m_i <path_to_model>/instance-segmentation-security-0228.xml
```

## Demo Output

The demo outputs original video with the processed one. Usage:

* Invert colors on the resulting frame by pressing the `i` key.
* Select a part of the frame to be shown in a separate window by using your left mouse button.
* Exit the demo by pressing `Esc`.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
