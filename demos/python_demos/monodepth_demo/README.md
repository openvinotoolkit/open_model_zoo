# MonoDepth Python Demo

This topic demonstrates how to run the MonoDepth demo application, which produces a disparity map for a given input image.
To this end, the code uses the network described in the [paper](https://arxiv.org/abs/1907.01341):

> Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer  
Rene Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network and an image to the
Inference Engine plugin. When inference is done, the application outputs the disparity map in pfm and png format (color-coded).

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:

``` 
python3 monodepth_demo.py -h
```

The command yields the following usage message:

``` 
usage: monodepth_demo.py [-h] -m MODEL -i INPUT [-l CPU_EXTENSION] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  -i INPUT, --input INPUT
                        Required. Path to a input image file
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels implementations
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. Sample will
                        look for a suitable plugin for device specified.
                        Default value is CPU
```

Running the application with the empty list of options yields the usage message given above and an error message.

## Demo Output

The application outputs are the floating disparity map (pfm) as well as a color-coded version (png).

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
