# TensorFlow* Object Detection Mask R-CNNs Segmentation C++ Demo

This topic demonstrates how to run the Segmentation demo application, which does inference using image segmentation networks created with Object Detection API.

The demo has a post-processing part that gathers mask arrays corresponding to bounding boxes with high probability taken from the Detection Output layer. Then the demo produces pictures with identified masks.

## How It Works

On startup, the demo application reads command line parameters and loads a network and an image to the Inference Engine plugin. When inference is done, the application creates an output image.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/mask_rcnn_demo/cpp/models.lst` file.
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

* mask_rcnn_inception_resnet_v2_atrous_coco
* mask_rcnn_inception_v2_coco
* mask_rcnn_resnet101_atrous_coco
* mask_rcnn_resnet50_atrous_coco

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

mask_rcnn_demo [OPTION]
Options:

    -h                                Print a usage message.
    -i "<path>"                       Required. Path to a .bmp image.
    -m "<path>"                       Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"            Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
          Or
      -c "<absolute_path>"            Required for GPU custom kernels. Absolute path to the .xml file with the kernels descriptions.
    -d "<device>"                     Optional. Specify the target device to infer on (the list of available devices is shown below). Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device (CPU by default)
    -detection_output_name "<string>" Optional. The name of detection output layer. Default value is "reshape_do_2d"
    -masks_name "<string>"            Optional. The name of masks layer. Default value is "masks"
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can use the following command to do inference on CPU on an image using a trained network:

```sh
./mask_rcnn_demo -i <path_to_image>/inputImage.bmp -m <path_to_model>/mask_rcnn_inception_resnet_v2_atrous_coco.xml
```

## Demo Output

For each input image the application outputs a segmented image. For example, `out0.png` and `out1.png` are created for the network with batch size equal to 2.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
