# Image Retrieval Python\* Demo

![example](./image_retrieval.gif)

This demo demonstrates how to run Image Retrieval models using OpenVINO&trade;.

> **NOTE**: Only batch size of 1 is supported.

## How It Works

The demo application expects an image retrieval model in the Intermediate Representation (IR) format.

As input, the demo application takes:

* a path to a list of images represented by textfile with following format: 'path_to_image' 'ID'
* a path to a video file or a device node of a webcam

The demo workflow is the following:

1. The demo application reads video frames one by one, runs ROI detector that extracts ROI (moving area).
2. Extracted ROI is passed to artificial neural network that computes embedding vector for extracted frame area.
3. Then the demo application searches computed embedding in gallery of images in order to determine which image in the gallery is the most similar to what one can see on video frame.
4. The app visualizes results of it work as graphical window where following objects are shown.
     - Input frame with detected ROI.
     - Top-10 most similar images from the gallery.
     - Performance characteristics.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

The demo sample input videos and gallery images can be found in this [repository](https://github.com/19900531/test). An example of file listing gallery images can be found [here](https://github.com/openvinotoolkit/training_extensions/blob/develop/misc/tensorflow_toolkit/image_retrieval/data/gallery/gallery.txt).

The demo dependencies should be installed before run. That can be achieved with the following command:

```sh
python3 -mpip install --user -r <omz_dir>/demos/requirements.txt
```

The list of models supported by the demo is in `<omz_dir>/demos/image_retrieval_demo/python/models.lst` file.
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

* image-retrieval-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: image_retrieval_demo.py [-h] -m MODEL -i INPUT [--loop]
                               [-o OUTPUT] [-limit OUTPUT_LIMIT]
                               -g GALLERY [-gt GROUND_TRUTH]
                               [-d DEVICE] [-l CPU_EXTENSION]
                               [--no_show] [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to a video file or a device node of a
                        webcam.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output file/s to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  -g GALLERY, --gallery GALLERY
                        Required. Path to a file listing gallery images.
  -gt GROUND_TRUTH, --ground_truth GROUND_TRUTH
                        Optional. Ground truth class.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on: CPU,
                        GPU, FPGA, HDDL or MYRIAD. The demo will look for a
                        suitable plugin for device specified (by default, it
                        is CPU).
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  --no_show             Optional. Do not visualize inference results.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, please provide paths to the model in the IR format, to a file with class labels, and to an input video, image, or folder with images:

```bash
python image_retrieval_demo.py \
-m <path_to_model>/image-retrieval-0001.xml \
-i <path_to_video>/4946fb41-9da0-4af7-a858-b443bee6d0f6.dav \
-g <path_to_file>/list.txt \
--ground_truth text_label
```

When single image applied as an input, the demo will process and render it quickly, then exit. In this particular case, recommendation is to also apply `loop` option, which will enforce looping over processing the single image, so processed results will be continuously visualized on screen.
The demo allow saving of processed results to a Motion JPEG AVI file or separate JPEG or PNG files when `-o` option is used. To save processed results in AVI file, the name of output file with `avi` extension should be specified with `-o` option, for example: `-o output.avi`. To save processed results as an images, the template name of output image file with `jpg` or `png` extension should be specified with `-o` option, as shown on example: `-o output_%03d.jpg`. The actual file names will be constructed from template at runtime by replacing regular expression `%03d` with frame number, resulting in storing files with names like following: `output_000.jpg`, `output_001.jpg`, and so on.
Amount of data stored in output file or files could be limited with `limit` option, to avoid disk space overrun in case of continouus input stream, like camera. Default value is 1000, it can be changed by applying `-limit N` option, where N is number of frames to store.
In case folder of pictures is used as a demo input the recommendation is to store results as images too, storing to AVI file may not work if input images are differs in resolution.

>**NOTE**: Windows* systems may not have Motion JPEG codec installed by default. If this is the case, OpenCV FFMPEG backend could be downloaded by PowerShell script, located at OpenVINO install package at the path `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. This script should be run with Administrative privileges. Or, alternatively, storing results to images can be used.

## Demo Output

The application uses OpenCV to display gallery searching result and current inference performance.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
