# Classification Python\* Demo

![](./classification.gif)

This demo showcases inference of Classification networks using Python\* Model API and Async Pipeline.

## How It Works

On startup, the application reads command line parameters and loads a classification network to the Inference Engine for execution. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

You can stop the demo by pressing "Esc" or "Q" button. After that, the average metrics values will be printed to the console.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html#general-conversion-parameters).

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/classification_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models

* alexnet
* caffenet
* densenet-121
* densenet-121-tf
* dla-34
* efficientnet-b0
* efficientnet-b0-pytorch
* efficientnet-v2-b0
* efficientnet-v2-s
* googlenet-v1
* googlenet-v1-tf
* googlenet-v2
* googlenet-v3
* googlenet-v3-pytorch
* googlenet-v4-tf
* hbonet-0.25
* hbonet-1.0
* inception-resnet-v2-tf
* mixnet-l
* mobilenet-v1-0.25-128
* mobilenet-v1-1.0-224
* mobilenet-v1-1.0-224-tf
* mobilenet-v2
* mobilenet-v2-1.0-224
* mobilenet-v2-1.4-224
* mobilenet-v2-pytorch
* nfnet-f0
* octave-resnet-26-0.25
* regnetx-3.2gf
* repvgg-a0
* repvgg-b1
* repvgg-b3
* resnest-50-pytorch
* resnet-18-pytorch
* resnet-50-pytorch
* resnet-50-tf
* resnet18-xnor-binary-onnx-0001
* resnet50-binary-0001
* rexnet-v1-x1.0
* se-inception
* se-resnet-50
* se-resnext-50
* shufflenet-v2-x0.5
* shufflenet-v2-x1.0
* squeezenet1.0
* squeezenet1.1
* swin-tiny-patch4-window7-224
* vgg16
* vgg19

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

### Required Files

If you want to see classification results, you must use "-labels" flags to specify .txt file containing lists of classes and labels.

Please note that you should use `<omz_dir>/data/dataset_classes/imagenet_2015.txt` labels file with the following models:

* googlenet-v2
* se-inception
* se-resnet-50
* se-resnext-50

and `<omz_dir>/data/dataset_classes/imagenet_2012.txt` labels file with all other models supported by the demo.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: classification_demo.py [-h] -m MODEL [--adapter {openvino,ovms}] -i INPUT
                              [-d DEVICE] [--labels LABELS]
                              [-topk {1,2,3,4,5,6,7,8,9,10}]
                              [-nireq NUM_INFER_REQUESTS]
                              [-nstreams NUM_STREAMS] [-nthreads NUM_THREADS]
                              [--loop] [-o OUTPUT] [-limit OUTPUT_LIMIT]
                              [--no_show]
                              [--output_resolution OUTPUT_RESOLUTION]
                              [-u UTILIZATION_MONITORS]
                              [--reverse_input_channels]
                              [--mean_values MEAN_VALUES MEAN_VALUES MEAN_VALUES]
                              [--scale_values SCALE_VALUES SCALE_VALUES SCALE_VALUES]
                              [-r]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  --adapter {openvino,ovms}
                        Optional. Specify the model adapter. Default is
                        openvino.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera
                        id.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, HDDL or MYRIAD is acceptable. The demo will look
                        for a suitable plugin for device specified. Default
                        value is CPU.

Common model options:
  --labels LABELS       Optional. Labels mapping file.
  -topk {1,2,3,4,5,6,7,8,9,10}
                        Optional. Number of top results. Default value is 5.
                        Must be from 1 to 10.

Inference options:
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests
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
                        Optional. Name of the output file(s) to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output. If 0 is
                        set, all frames are stored.
  --no_show             Optional. Don't show output.
  --output_resolution OUTPUT_RESOLUTION
                        Optional. Specify the maximum output window resolution
                        in (width x height) format. Example: 1280x720. Input
                        frame size used by default.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.

Input transform options:
  --reverse_input_channels
                        Optional. Switch the input channels order from BGR to
                        RGB.
  --mean_values MEAN_VALUES MEAN_VALUES MEAN_VALUES
                        Optional. Normalize input by subtracting the mean
                        values per channel. Example: 255.0 255.0 255.0
  --scale_values SCALE_VALUES SCALE_VALUES SCALE_VALUES
                        Optional. Divide input by scale values per channel.
                        Division is applied after mean values subtraction.
                        Example: 255.0 255.0 255.0

Debug options:
  -r, --raw_output_message
                        Optional. Output inference results raw values showing.
```

Running the application with the empty list of options yields an error message.

For example, use the following command-line command to run the application:

```sh
python3 classification_demo.py -m <path_to_classification_model> \
                               -i <path_to_folder_with_images> \
                                --labels <path_to_file_with_list_of_labels>
```

## Demo Output

The demo uses OpenCV to display images with classification results presented as a text on them. The demo reports:

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).
* Latency for each of the following pipeline stages:
  * **Decoding** — capturing input data.
  * **Preprocessing** — data preparation for inference.
  * **Inference** — infering input data (images) and getting a result.
  * **Postrocessing** — preparation inference result for output.
  * **Rendering** — generating output image.

You can use these metrics to measure application-level performance.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
