# Colorization Demo

This demo demonstrates an example of using neural networks to colorize a grayscale image or video.

## How It Works

On startup, the application reads command-line parameters and loads one network to the Inference Engine for execution. Once the program receives an image, it performs the following steps:

1. Converts the frame into the LAB color space.
2. Uses the L-channel to predict A and B channels.
3. Restores the image by converting it into the BGR color space.

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/colorization_demo/python/models.lst` file.
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

* colorization-v2
* colorization-siggraph

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running the Demo

Running the application with the `-h` option yields the following usage message:

```
usage: colorization_demo.py [-h] -m MODEL [-d DEVICE] -i INPUT [--loop]
                            [-o OUTPUT] [-limit OUTPUT_LIMIT]
                            [--no_show] [-v] [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Help with the script.
  -m MODEL, --model MODEL
                        Required. Path to .xml file with pre-trained model.
  -d DEVICE, --device DEVICE
                        Optional. Specify target device for infer: CPU, GPU,
                        FPGA, HDDL or MYRIAD. Default: CPU
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a single image,
                        a folder of images, video file or camera id.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output file/s to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  --no_show             Optional. Don't show output.
  -v, --verbose         Optional. Enable display of processing logs on screen.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```
Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, please provide paths to the model in the IR format, and to an input video or image(s):

```bash
python colorization_demo.py \
    -i <path_to_image>/<image_name>.jpg \
    -m <path_to_model>/colorization-v2.xml
```

When single image applied as an input, the demo will process and render it quickly, then exit. In this particular case, recommendation is to also apply `loop` option, which will enforce looping over processing the single image, so processed results will be continuously visualized on screen.
The demo allow saving of processed results to a Motion JPEG AVI file or separate JPEG or PNG files when `-o` option is used. To save processed results in AVI file, the name of output file with `avi` extension should be specified with `-o` option, for example: `-o output.avi`. To save processed results as an images, the template name of output image file with `jpg` or `png` extension should be specified with `-o` option, as shown on example: `-o output_%03d.jpg`. The actual file names will be constructed from template at runtime by replacing regular expression `%03d` with frame number, resulting in storing files with names like following: `output_000.jpg`, `output_001.jpg`, and so on.
Amount of data stored in output file or files could be limited with `limit` option, to avoid disk space overrun in case of continouus input stream, like camera. Default value is 1000, it can be changed by applying `-limit N` option, where N is number of frames to store.
In case folder of pictures is used as a demo input the recommendation is to store results as images too, storing to AVI file may not work if input images are differs in resolution.

>**NOTE**: Windows* systems may not have Motion JPEG codec installed by default. If this is the case, OpenCV FFMPEG backend could be downloaded by PowerShell script, located at OpenVINO install package at the path `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. This script should be run with Administrative privileges. Or, alternatively, storing results to images can be used.

## Demo Output

The demo uses OpenCV to display the colorized frame.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
