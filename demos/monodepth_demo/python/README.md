# MonoDepth Python Demo

This topic demonstrates how to run the MonoDepth demo application, which produces a disparity map for a given input image.
To this end, the code uses the network described in [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341).

Below is the `midasnet` model inference result:

![example](./disp.png)

## How It Works

On startup, the application reads command-line parameters and loads a model to OpenVINO™ Runtime plugin. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

Async API operates with a notion of the "Infer Request" that encapsulates the inputs/outputs and separates
*scheduling and waiting for result*.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).

## Model API

The demo utilizes model wrappers, adapters and pipelines from [Python* Model API](../../common/python/model_zoo/model_api/README.md).

The generalized interface of wrappers with its unified results representation provides the support of multiple different monocular depth estimation model topologies in one demo.

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/monodepth_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO IR format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models

* fcrn-dp-nyu-depth-v2-tf
* midasnet

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: monodepth_demo.py [-h] -m MODEL -i INPUT [-d DEVICE]
                         [--adapter {openvino,ovms}] [-nireq NUM_INFER_REQUESTS] [-nstreams NUM_STREAMS]
                         [-nthreads NUM_THREADS] [--loop] [-o OUTPUT] [-limit OUTPUT_LIMIT] [--no_show]
                         [--output_resolution OUTPUT_RESOLUTION] [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model or
                        address of model inference service if using OVMS adapter.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a single image, a folder of images, video
                        file or camera id.
  --adapter {openvino,ovms}
                        Optional. Specify the model adapter. Default is
                        openvino.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU or GPU is acceptable. The
                        demo will look for a suitable plugin for device specified. Default value is CPU.

Inference options:
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests.
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode (for
                        HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just
                        <nstreams>).
  -nthreads NUM_THREADS, --num_threads NUM_THREADS
                        Optional. Number of threads to use for inference on CPU (including HETERO cases).

Input/output options:
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output. If 0 is set, all frames are stored.
  --no_show             Optional. Don't show output.
  --output_resolution OUTPUT_RESOLUTION
                        Optional. Specify the maximum output window resolution in (width x height) format. Example:
                        1280x720. Input frame size used by default.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can use the following command to do inference on GPU with a pre-trained midasnet model:

```sh
python3 monodepth_demo.py \
  -d GPU \
  -i <path_to_video>/inputVideo.mp4 \
  -m <path_to_model>/midasnet.xml
```

The number of Infer Requests is specified by `-nireq` flag. An increase of this number usually leads to an increase
of performance (throughput), since in this case several Infer Requests can be processed simultaneously if the device
supports parallelization. However, a large number of Infer Requests increases the latency because each frame still
has to wait before being sent for inference.

For higher FPS, it is recommended that you set `-nireq` to slightly exceed the `-nstreams` value,
summed across all devices used.

> **NOTE**: This demo is based on the callback functionality from the OpenVINO™ Runtime API.
  The selected approach makes the execution in multi-device mode optimal by preventing wait delays caused by
  the differences in device performance. However, the internal organization of the callback mechanism in Python API
  leads to a decrease in FPS.

>**NOTE**: If you provide a single image as an input, the demo processes and renders it quickly, then exits. To continuously visualize inference results on the screen, apply the `loop` option, which enforces processing a single image in a loop.

You can save processed results to a Motion JPEG AVI file or separate JPEG or PNG files using the `-o` option:

* To save processed results in an AVI file, specify the name of the output file with `avi` extension, for example: `-o output.avi`.
* To save processed results as images, specify the template name of the output image file with `jpg` or `png` extension, for example: `-o output_%03d.jpg`. The actual file names are constructed from the template at runtime by replacing regular expression `%03d` with the frame number, resulting in the following: `output_000.jpg`, `output_001.jpg`, and so on.
To avoid disk space overrun in case of continuous input stream, like camera, you can limit the amount of data stored in the output file(s) with the `limit` option. The default value is 1000. To change it, you can apply the `-limit N` option, where `N` is the number of frames to store.

>**NOTE**: Windows\* systems may not have the Motion JPEG codec installed by default. If this is the case, you can download OpenCV FFMPEG back end using the PowerShell script provided with the OpenVINO &trade; install package and located at `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. The script should be run with administrative privileges if OpenVINO &trade; is installed in a system protected folder (this is a typical case). Alternatively, you can save results as images.

## Running with OpenVINO Model Server

You can also run this demo with model served in [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server). Refer to [`OVMSAdapter`](../../common/python/model_zoo/model_api/adapters/ovms_adapter.md) to learn about running demos with OVMS.

Exemplary command:

```sh
python3 monodepth_demo.py \
  -i <path_to_video>/inputVideo.mp4 \
  -m localhost:9000/models/monodepth \
  --adapter ovms
```

## Demo Output

The demo uses OpenCV to display the resulting frame with colored depth map.
The demo reports:

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).
You can use both of these metrics to measure application-level performance.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
