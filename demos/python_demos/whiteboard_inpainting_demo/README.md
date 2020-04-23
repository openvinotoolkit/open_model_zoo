# Whiteboard inpainting demo

This demo demonstrates how to disable a person from a whiteboard and show
text on the whiteboard using OpenVINO<sup>TM</sup>.

## How it works

The demo expects one of the next models in the Intermediate Representation (IR) format:

    * Instance segmentation model
    * Semantic segmentation model

It can be your own models or pre-trained model from OpenVINO Open Model Zoo.
In the `models.lst` are the list of appropriate models for this demo that can
be obtained via `Model downloader`. Please see more information about
`Model downloader` [here](../../../tools/downloader/README.md).

As input, the demo application takes:

    * paths to a video file
    * index of a web camera

## Running

### Installation of dependencies

To install required dependencies run

```bash
pip3 install -r requirements.txt
```

### Command line arguments

Run the application with the `-h` option to see the following usage message:

```
usage: whiteboard_inpainting_demo.py [-h] -i I [-mi M_INSTANCE_SEGMENTATION]
                                     [-ms M_SEMANTIC_SEGMENTATION] [-t THRESHOLD]
                                     [--output_video OUTPUT_VIDEO] [--no_show] [-d DEVICE]
                                     [-l CPU_EXTENSION]

Whiteboard inpainting demo

optional arguments:
  -h, --help            show this help message and exit
  -i I                  Input sources (index of camera or path to video file)
  -mi M_INSTANCE_SEGMENTATION, --m_instance_segmentation M_INSTANCE_SEGMENTATION
                        Path to the instance segmentation model
  -ms M_SEMANTIC_SEGMENTATION, --m_semantic_segmentation M_SEMANTIC_SEGMENTATION
                        Path to the semantic segmentation model
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold for person instance segmentation model
  --output_video OUTPUT_VIDEO
                        Optional. Path to output video
  --no_show             Optional. Don't show output
  -d DEVICE, --device DEVICE
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with the kernels impl.
```

Example of command:

```
python whiteboard_inpainting_demo.py \
    -i path/to/video.avi \
    -mi path/to/instance-segmentation-security-0050.xml \
```

## Using

During the demo execution it is possible to invert colors on the result frame
by pressing `i` key. Via mouse you can set a part of frame which will be shown
in a separate window (using left buttom of mouse). `Esc` - exit the demo.

