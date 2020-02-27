# Super Resolution C++ Demo

This topic demonstrates how to run Super Resolution demo application, which
reconstructs the high resolution image from the original low resolution one.
You can use the following pre-trained model with the demo:

* `single-image-super-resolution-1032`, which is the model that performs super resolution 4x upscale on a 270x480 image
* `single-image-super-resolution-1033`, which is the model that performs super resolution 3x upscale on a 360x640 image
* `text-image-super-resolution-0001`, which is the model that performs super resolution 3x upscale on a 360x640 image

For more information about the pre-trained models, refer to the [model documentation](../../models/intel/index.md).

## How It Works

On the start-up, the application reads command-line parameters and loads the
specified network. After that, the application reads an input image and
performs upscale using super resolution model.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

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

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

To do inference on CPU using a trained model, run the following command:

```sh
./super_resolution_demo -i <path_to_image>/image.bmp -m <path_to_model>/model.xml
```

## Demo Output

The application outputs a reconstructed high-resolution image and saves it in
the current working directory as `*.bmp` file with `sr` prefix.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo is not supported with any of the Model Downloader available topologies. Other models may produce unexpected results on these devices as well.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
