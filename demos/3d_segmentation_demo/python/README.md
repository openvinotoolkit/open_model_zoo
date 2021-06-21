# 3D Segmentation Python\* Demo

This topic demonstrates how to run the 3D Segmentation Demo, which segments 3D images using 3D convolutional networks.

## How It Works

On startup, the demo reads command-line parameters and loads a network and images to the Inference Engine plugin.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

The demo dependencies should be installed before run. That can be achieved with the following command:

```sh
python3 -mpip install --user -r <omz_dir>/demos/3d_segmentation_demo/python/requirements.txt
```

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/3d_segmentation_demo/python/models.lst` file.
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

* brain-tumor-segmentation-0001
* brain-tumor-segmentation-0002

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Run the application with the `-h` or `--help` option to see the usage message:

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
                        Required. Path to an input folder with NIfTI
                        data/NIFTI file/TIFF file
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
  -ms N1,N2,N3,N4, --mri_sequence N1,N2,N3,N4
                        Optional. Transfer MRI-sequence from dataset order to the network order.
  --full_intensities_range
                        Optional. Take intensities of the input image in a full range.
```

Running the application with the empty list of options yields the usage message and an error message.

For example, to do inference on a 3D TIFF image using a trained network with multiple outputs on CPU, run the following
command:

```sh
python3 3d_segmentation_demo.py -i <path_to_image>/inputImage.tiff -m <path_to_model>/brain-tumor-segmentation-0001.xml -d CPU -o <path_to_output>
```

For example, to do inference on 3D NIfTI images using a trained network with multiple outputs on CPU and save
output TIFF and NIFTI images, run the following command:

```sh
python3 3d_segmentation_demo.py -i <path_to_nifti_images> -m <path_to_model>/brain-tumor-segmentation-0001 -d CPU -o <path_to_output> -nii -ms 2,0,3,1
```

For example, to do inference on a single 3D NIfTI image and save an output TIFF image, run the following command:

```sh
python3 3d_segmentation_demo.py -i <path_to_nifti_image>/PackedImage.nii -m <path_to_model>/brain-tumor-segmentation-0001 -d CPU -o <path_to_output> -ms 2,0,3,1
```

`-ms` option aligns input modalities that depend on a dataset. For example, [Medical Decathlon](http://medicaldecathlon.com/) brain tumor segmentation data modalities follow in different order than it's required by nets. To make a correct order using Medical Decathlon brain tumor data the correct option is `2,0,3,1` for `brain-tumor-segmentation-0001` and `1,2,3,0` for `brain-tumor-segmentation-0002`.

```sh
python3 3d_segmentation_demo.py -i <path_to_nifti_images> -m <path_to_model>/brain-tumor-segmentation-0002 -d CPU -o <path_to_output> -nii -ms 1,2,3,0 --full_intensities_range
```

`--full_intensities_range` option is related to preprocessing of input data. It can be different for different models, for example, `brain-tumor-segmentation-0001` expects normalized data in [0,1] range and nullified non-positive values, while `brain-tumor-segmentation-0002` just requires z-score normalization in a full range. So to use `brain-tumor-segmentation-0002` model, the flag `--full_intensities_range` should be set, while for `brain-tumor-segmentation-0001` no preprocessing option is required.

## Demo Output

The demo outputs a multipage TIFF image and a NIFTI archive.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
