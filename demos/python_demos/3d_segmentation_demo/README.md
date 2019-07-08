# 3D Segmentation Python* Demo

This topic demonstrates how to run the 3D Segmentation Demo, which segments 3D images using 3D convolutional networks.

## How It Works

Upon the start-up, the demo reads command-line parameters and loads a network and images to the Inference Engine plugin.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Run the application with the `-h` or `--help` option to see the usage message:
```
python3 3d_segmentation_demo.py -h
```
The command yields the following usage message:
```
usage: 3d_segmentation_demo.py [-h] -i PATH_TO_INPUT_DATA -m PATH_TO_MODEL -o
                               PATH_TO_OUTPUT [-d TARGET_DEVICE]
                               [-l PATH_TO_EXTENSION] [-nii]
                               [-nthreads NUMBER_THREADS]
                               [-s [SHAPE [SHAPE ...]]]
                               [-c PATH_TO_CLDNN_CONFIG]

Options:
  -h, --help            Show this help message and exit.
  -i PATH_TO_INPUT_DATA, --path_to_input_data PATH_TO_INPUT_DATA
                        Required. Path to an input folder with NIfTI data/TIFF
                        file
  -m PATH_TO_MODEL, --path_to_model PATH_TO_MODEL
                        Required. Path to an .xml file with a trained model
  -o PATH_TO_OUTPUT, --path_to_output PATH_TO_OUTPUT
                        Required. Path to a folder where output files will be
                        saved
  -d TARGET_DEVICE, --target_device TARGET_DEVICE
                        Optional. Specify a target device to infer on: CPU, GPU.
                        Use "-d HETERO:<comma separated devices list>" format
                        to specify HETERO plugin.
  -l PATH_TO_EXTENSION, --path_to_extension PATH_TO_EXTENSION
                        Required for CPU custom layers. Absolute path to a
                        shared library with the kernels implementations.
  -nii, --output_nifti  Show output inference results as raw values
  -nthreads NUMBER_THREADS, --number_threads NUMBER_THREADS
                        Optional. Number of threads to use for inference on
                        CPU (including HETERO cases).
  -s [SHAPE [SHAPE ...]], --shape [SHAPE [SHAPE ...]]
                        Optional. Specify shape for a network
  -c PATH_TO_CLDNN_CONFIG, --path_to_cldnn_config PATH_TO_CLDNN_CONFIG
                        Required for GPU custom kernels. Absolute path to an
                        .xml file with the kernels description.
```

Running the application with the empty list of options yields the usage message and an error message.
To run the demo, use public or pre-trained models that support 3D convolution, for example, UNet3D. You can download the pre-trained models using the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to do inference on a 3D TIFF image using a trained network with multiple outputs on CPU, run the following 
command:

```
python3 3d_segmentation_demo.py -i <path_to_image>/inputImage.tiff -m <path_to_model>/multiple-output.xml -d CPU -o <path_to_output>
```
     
For example, to do inference on an 3D NIfTI image using a trained network with multiple outputs on CPU and save 
output TIFF and NIFTI images, run the following command:
```
python3 3d_segmentation_demo.py -i <path_to_nifti_images> -m <path_to_model>/multiple-output.xml -d CPU -o <path_to_output> -nii
```
     
## Demo Output
The demo outputs a multipage TIFF image and a NIFTI archive.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
