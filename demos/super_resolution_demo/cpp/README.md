# Super Resolution C++ Demo

This topic demonstrates how to run Super Resolution demo application, which
reconstructs the high resolution image from the original low resolution one.
You can use the following pre-trained model with the demo:

* `single-image-super-resolution-1032`, which is the model that performs super resolution 4x upscale on a 270x480 image
* `single-image-super-resolution-1033`, which is the model that performs super resolution 3x upscale on a 360x640 image
* `text-image-super-resolution-0001`, which is the model that performs super resolution 3x upscale on a 360x640 image

## How It Works

On the start-up, the application reads command-line parameters and loads the
specified network. After that, the application reads an input image and
performs upscale using super resolution model.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in <omz_dir>/demos/super_resolution_demo/cpp/models.lst file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported Models

* single-image-super-resolution-1032
* single-image-super-resolution-1033
* text-image-super-resolution-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
./super_resolution_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

super_resolution_demo [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Path to an image.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for the specified device.
    -show                   Optional. Show processed images. Default value is false.

```

Running the application with the empty list of options yields the usage message given above and an error message.

To do inference on CPU using a trained model, run the following command:

```sh
./super_resolution_demo -d CPU -i <path_to_image>/image.bmp -m <path_to_model>/single-image-super-resolution-1032.xml
```

## Demo Output

The application outputs a reconstructed high-resolution image and saves it in the current working directory as `*.bmp` file with `sr` prefix.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
