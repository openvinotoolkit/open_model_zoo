# TensorFlow* Object Detection Mask R-CNNs Segmentation C++ Demo

This topic demonstrates how to run the Segmentation demo application, which does inference using image segmentation networks created with Object Detection API.

The demo has a post-processing part that gathers masks arrays corresponding to bounding boxes with high probability taken from the Detection Output layer. Then the demo produces pictures with identified masks.

## How It Works

Upon the start-up, the demo application reads command line parameters and loads a network and an image to the Inference Engine plugin. When inference is done, the application creates an output image.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```
./mask_rcnn_demo -h
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

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

You can use the following command to do inference on CPU on an image using a trained network:
```sh
./mask_rcnn_demo -i <path_to_image>/inputImage.bmp -m <path_to_model>/mask_rcnn_inception_resnet_v2_atrous_coco.xml
```

## Demo Output

For each input image the application outputs a segmented image. For example, `out0.png` and `out1.png` are created for the network with batch size equal to 2.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo is not supported with any of the Model Downloader available topologies. Other models may produce unexpected results on these devices as well.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
