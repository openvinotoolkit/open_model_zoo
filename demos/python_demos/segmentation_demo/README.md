# Image Segmentation Python* Demo

This topic demonstrates how to run the Image Segmentation demo application, which does inference using image
segmentation networks like FCN8.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network and an image to the
Inference Engine plugin. When inference is done, the application creates an output image.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```
python3 segmentation_demo.py -h
```
The command yields the following usage message:
```
usage: segmentation_demo.py [-h] -m MODEL -i INPUT [INPUT ...]
                            [-l CPU_EXTENSION] [-d DEVICE]
                            [-nt NUMBER_TOP] [-ni NUMBER_ITER] [-pc]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Required. Path to a folder with images or path to an
                        image files
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels implementations
  -d DEVICE, --device DEVICE
                        Optional. Required for CPU custom layers Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. Sample will
                        look for a suitable plugin for device specified (CPU
                        by default)
  -nt NUMBER_TOP, --number_top NUMBER_TOP
                        Optional. Number of top results
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).


You can use the following command do inference on Intel&reg CPU; Processors on an image using a trained FCN8 network:
```
    python3 segmentation_demo.py -i <path_to_image>/inputImage.bmp -m <path_to_model>/fcn8.xml
```

## Demo Output

The application outputs are a segmented image (`out.bmp`).


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
