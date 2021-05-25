# Image Segmentation Python\* Demo

![example](../segmentation.gif)

This topic demonstrates how to run the Image Segmentation demo application, which does inference using semantic segmentation networks.

> **NOTE:** This topic describes usage of Python\* implementation of the Image Segmentation Demo. For the C++ implementation, refer to [Image Segmentation C++ Demo](../cpp/README.md).

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network. The demo runs inference and shows results for each image captured from an input. Demo provides default mapping of classes to colors and optionally, allow to specify mapping of classes to colors from simple text file, with using `--colors` argument. Depending on number of inference requests processing simultaneously (-nireq parameter) the pipeline might minimize the time required to process each single image (for nireq 1) or maximize utilization of the device and overall processing performance.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/segmentation_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported Models

* architecture_type = segmentation
  - deeplabv3
  - fastseg-large
  - fastseg-small
  - hrnet-v2-c1-segmentation
  - icnet-camvid-ava-0001
  - icnet-camvid-ava-sparse-30-0001
  - icnet-camvid-ava-sparse-60-0001
  - pspnet-pytorch
  - road-segmentation-adas-0001
  - semantic-segmentation-adas-0001
  - unet-camvid-onnx-0001
* architecture_type = salient_object_detection
  - f3net

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: segmentation_demo.py [-h] -m MODEL -i INPUT
                            [-at {segmentation,salient_object_detection}
                            [-d DEVICE] [-c COLORS]
                            [-nireq NUM_INFER_REQUESTS]
                            [-nstreams NUM_STREAMS]
                            [-nthreads NUM_THREADS]
                            [--loop] [-o OUTPUT]
                            [-limit OUTPUT_LIMIT] [--no_show]
                            [--output_resolution OUTPUT_RESOLUTION]
                            [-u UTILIZATION_MONITORS]
Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -at {segmentation, salient_object_detection}, --architecture_type {segmentation, salient_object_detection}
                        Optional. Default value is segmentation. Specify model's architecture type.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera id.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The demo
                        will look for a suitable plugin for device specified.
                        Default value is CPU.

Common model options:
  -c COLORS, --colors COLORS
                        Optional. Path to a text file containing colors for
                        classes.

Inference options:
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests.
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        Optional. Number of streams to use for inference on
                        the CPU or/and GPU in throughput mode (for HETERO and
                        MULTI device cases use format
                        <device1>:<nstreams1>,<device2>:<nstreams2> or just
                        <nstreams>).
  -nthreads NUM_THREADS, --num_threads NUM_THREADS
                        Optional. Number of threads to use for inference on
                        CPU (including HETERO cases).

Input/output options:
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  --no_show             Optional. Don't show output.
  --output_resolution OUTPUT_RESOLUTION
                        Optional. Specify the maximum output window resolution
                        in (width x height) format. Example: 1280x720.
                        Input frame size used by default.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can use the following command to do inference on CPU on images captured by a camera using a pre-trained network:

```sh
    python3 segmentation_demo.py -d CPU -i 0 -m <path_to_model>/semantic-segmentation-adas-0001.xml
```

## Color Palettes

The color palette is used to visualize predicted classes. By default, the colors from PASCAL VOC dataset are applied. In case then the number of output classes is larger than number of classes provided by PASCAL VOC dataset, the rest classes are randomly colorized. Also, one can use predefined colors from other datasets, like CAMVID.

Available colors files located at `<omz_dir>/data/palettes` folder. If you want to assign custom colors for classes, you should create a `.txt` file, where the each line contains colors in `(R, G, B)` format. The demo application treat the number of each line as a dataset class identificator and apply specified color to pixels belonging to this class.

## Demo Output

The demo uses OpenCV to display the resulting images with blended segmentation mask.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
