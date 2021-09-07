# Image Processing C++ Demo

This demo processes the image according to the selected type of processing. The demo can work with the next types:

* `super_resolution`
* `deblurring`

## Examples

Exmaple for deblurring type (left - source image, right - image after deblurring):

![](./assets/deblurred_image.png)

All images on result frame will be marked one of these flags:

* 'O' - original image.
* 'R' - result image.
* 'D' - difference image (|result - original|).

Example for super_resolution type:

Low resolution:

![](./assets/street_640x360.png)

Bicubic interpolation:

![](./assets/street_resized.png)

Super resolution:

![](./assets/street_resolution.png)

## How It Works

Before running the demo, user must choose type of processing and model for this processing.
For `super_resolution` user can choose the next models:

* [single-image-super-resolution-1032](../../../models/intel/single-image-super-resolution-1032/README.md) enhances the resolution of the input image by a factor of 4.
* [single-image-super-resolution-1033](../../../models/intel/single-image-super-resolution-1033/README.md) enhances the resolution of the input image by a factor of 3.
* [text-image-super-resolution-0001](../../../models/intel/text-image-super-resolution-0001/README.md) - a tiny model to 3x upscale scanned images with text.

For `deblurring` user can use [deblurgan-v2](../../../models/public/deblurgan-v2/README.md) - generative adversarial network for single image motion deblurring.

The demo runs inference and shows results for each image captured from an input. Depending on number of inference requests processing simultaneously (-nireq parameter) the pipeline might minimize the time required to process each single image (for nireq 1) or maximizes utilization of the device and overall processing performance.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/segmentation_demo/cpp/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported Models

* single-image-super-resolution-1032
* single-image-super-resolution-1033
* text-image-super-resolution-0001
* deblurgan-v2

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
[ INFO ] InferenceEngine: <version>

image_processing_demo_async [OPTION]
Options:

    -h                        Print a usage message.
    -at "<type>"              Required. Type of the network, either 'sr' for Super Resolution task or 'deblur' for Deblurring
    -i "<path>"               Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -m "<path>"               Required. Path to an .xml file with a trained model.
    -o "<path>"               Optional. Name of the output file(s) to save.
    -limit "<num>"            Optional. Number of frames to store in output. If 0 is set, all frames are stored.
      -l "<absolute_path>"    Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
          Or
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with the kernel descriptions.
    -d "<device>"             Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -pc                       Optional. Enables per-layer performance report.
    -nireq "<integer>"        Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.
    -nthreads "<integer>"     Optional. Number of threads.
    -nstreams                 Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)
    -loop                     Optional. Enable reading the input in a loop.
    -no_show                  Optional. Do not show processed video.
    -output_resolution        Optional. Specify the maximum output window resolution in (width x height) format. Example: 1280x720. Input frame size used by default.
    -u                        Optional. List of monitors to show initially.
```

Running the application with the empty list of options yields an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in [models.lst](./models.lst).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

You can use the following command to enhance the resolution of the images captured by a camera using a pre-trained single-image-super-resolution-1033 network:

```sh
./image_processing_demo -i 0 -m single-image-super-resolution-1033.xml -at sr
```

### Modes

Demo application supports 3 modes:

1. to display the result image.
2. to display the original image with result together (left part is result, right part is original). Position of separator is specified be slider of trackbar.
3. to display the difference image with result together. By analogy with second mode.

User is able to change mode in run time using the next keys:

* **R** to display the result.
* **O** to display the original image with result.
* **V** to display the difference image with result.
* **Esc or Q** to quit

You can save processed results to a Motion JPEG AVI file or separate JPEG or PNG files using the `-o` option:

* To save processed results in an AVI file, specify the name of the output file with `avi` extension, for example: `-o output.avi`.
* To save processed results as images, specify the template name of the output image file with `jpg` or `png` extension, for example: `-o output_%03d.jpg`. The actual file names are constructed from the template at runtime by replacing regular expression `%03d` with the frame number, resulting in the following: `output_000.jpg`, `output_001.jpg`, and so on.
To avoid disk space overrun in case of continuous input stream, like camera, you can limit the amount of data stored in the output file(s) with the `limit` option. The default value is 1000. To change it, you can apply the `-limit N` option, where `N` is the number of frames to store.

>**NOTE**: Windows\* systems may not have the Motion JPEG codec installed by default. If this is the case, you can download OpenCV FFMPEG back end using the PowerShell script provided with the OpenVINO &trade; install package and located at `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. The script should be run with administrative privileges if OpenVINO &trade; is installed in a system protected folder (this is a typical case). Alternatively, you can save results as images.

## Demo Output

The demo uses OpenCV to display and write the resulting images.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
