# Background subtraction Python\* Demo

![example](./background_subtraction.gif)

This demo shows how to perform background subtraction using OpenVINO.

> **NOTE**: Only batch size of 1 is supported.

## How It Works

The demo application expects an instance segmentation or background matting model in the Intermediate Representation (IR) format with the following constraints:
1. for instance segmentation models based on `Mask RCNN` approach:
    * One input: `image` for input image.
    * At least three outputs including:
        * `boxes` with absolute bounding box coordinates of the input image and its score
        * `labels` with object class IDs for all bounding boxes
        * `masks` with fixed-size segmentation heat maps for all classes of all bounding boxes
2. for instance segmentation models based on `YOLACT` approach:
    * Single input for input image.
    * At least four outputs including:
        * `boxes` with normalized in [0, 1] range bounding box coordinates
        * `conf` with confidence scores for each class for all boxes
        * `mask` with fixed-size mask channels for all boxes.
        * `proto` with fixed-size segmentation heat maps prototypes for all boxes.
3. for image background matting models:
    * Two inputs:
        * `src` for input image
        * `bgr` for input real background
    * At least two outputs including:
        * `fgr` with normalized in [0, 1] range foreground
        * `pha` with normalized in [0, 1] range alpha
4. for image background matting models without trimap (background segmentation):
    * Single input for input image.
    * Single output with normalized in [0, 1] range alpha
5. for video background matting models based on RNN architecture:
    * Five inputs:
        * `src` for input image
        * recurrent inputs: `r1`, `r2`, `r3`, `r4`
    * At least six outputs including:
        * `fgr` with normalized in [0, 1] range foreground
        * `pha` with normalized in [0, 1] range alpha
        * recurrent outputs: `rr1`, `rr2`, `rr3`, `rr4`

The use case for the demo is an online conference where is needed to show only foreground - people and, respectively, to hide or replace background.
Based on this an instance segmentation model must be trained at least for person class.

As input, the demo application accepts a path to a single image file, a video file or a numeric ID of a web camera specified with a command-line argument `-i`

> **NOTE**: if you use image background matting models, `--background` argument should be specified. This is a background image that equal to a real background behind a person on an input frame and must have the same shape as an input image.

The demo workflow is the following:

1. The demo application reads image/video frames one by one, resizes them to fit into the input image blob of the network (`image`).
2. The demo visualizes the resulting background subtraction. Certain command-line options affect the visualization:
    * If you specify `--target_bgr`, background will be replaced by a chosen image or video. By default background replaced by green field.
    * If you specify `--blur_bgr`, background will be blurred according to a set value. By default equal to zero and is not applied.
    * If you specify `--show_with_original_frame`, the result image will be merged with an input one.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).

## Model API

The demo utilizes model wrappers, adapters and pipelines from [Python* Model API](../../common/python/model_zoo/model_api/README.md).

The generalized interface of wrappers with its unified results representation provides the support of multiple different background subtraction model topologies in one demo.

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/instance_segmentation_demo/python/models.lst` file.
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

* background-matting-mobilenetv2
* instance-segmentation-person-????
* modnet-photographic-portrait-matting
* modnet-webcam-portrait-matting
* robust-video-matting-mobilenetv3
* yolact-resnet50-fpn-pytorch

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: background_subtraction_demo.py [-h] -m MODEL [--adapter {openvino,ovms}] -i INPUT [-d DEVICE] [-t PROB_THRESHOLD] [--resize_type {crop,standard,fit_to_window,fit_to_window_letterbox}] [--labels LABELS] [--target_bgr TARGET_BGR]
                                      [--background BACKGROUND] [--blur_bgr BLUR_BGR] [--layout LAYOUT] [-nireq NUM_INFER_REQUESTS] [-nstreams NUM_STREAMS] [-nthreads NUM_THREADS] [--loop] [-o OUTPUT] [-limit OUTPUT_LIMIT] [--no_show]
                                      [--show_with_original_frame] [--output_resolution OUTPUT_RESOLUTION] [-u UTILIZATION_MONITORS] [-r]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model or address of model inference service if using OVMS adapter.
  --adapter {openvino,ovms}
                        Optional. Specify the model adapter. Default is openvino.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
  -d DEVICE, --device DEVICE
                        Optional. Specify a device to infer on (the list of available devices is shown below). Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use '-d MULTI:<comma-separated_devices_list>'
                        format to specify MULTI plugin. Default is CPU
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections filtering.
  --resize_type {crop,standard,fit_to_window,fit_to_window_letterbox}
                        Optional. A resize type for model preprocess. By default used model predefined type.
  --labels LABELS       Optional. Labels mapping file.
  --target_bgr TARGET_BGR
                        Optional. Background onto which to composite the output (by default to green field).
  --background BACKGROUND
                        Optional. Background image for background-matting model. This is a background image that equal to a real background behind a person on an input frame and must have the same shape as an input image.
  --blur_bgr BLUR_BGR   Optional. Background blur strength (by default with value 0 is not applied).
  --layout LAYOUT       Optional. Model inputs layouts. Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.

Inference options:
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests.
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).
  -nthreads NUM_THREADS, --num_threads NUM_THREADS
                        Optional. Number of threads to use for inference on CPU (including HETERO cases).

Input/output options:
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output. If 0 is set, all frames are stored.
  --no_show             Optional. Don't show output.
  --show_with_original_frame
                        Optional. Merge the result frame with the original one.
  --output_resolution OUTPUT_RESOLUTION
                        Optional. Specify the maximum output window resolution in (width x height) format. Example: 1280x720. Input frame size used by default.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.

Debug options:
  -r, --raw_output_message
                        Optional. Output inference results as mask histogram.
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, please provide paths to the model in the IR format, and to an input video, image, or folder with images:

```bash
python3 background_subtraction_demo/python/background_subtraction_demo.py \
    -m <path_to_model>/instance-segmentation-person-0007.xml \
    -i 0
```

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
python3 background_subtraction_demo/python/background_subtraction_demo.py \
    -m localhost:9000/models/background_subtraction \
    -i 0 \
    --adapter ovms
```

## Demo Output

The application uses OpenCV to display resulting images.
The demo reports

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
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
