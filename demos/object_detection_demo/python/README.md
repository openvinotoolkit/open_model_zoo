# Object Detection Python\* Demo

![example](../object_detection.gif)

This demo showcases inference of Object Detection networks using Sync and Async API.

Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the app can continue doing things on the host, while accelerator is busy.
Specifically, this demo keeps the number of Infer Requests that you have set using `-nireq` flag.
While some of the Infer Requests are processed by OpenVINO™ Runtime, the other ones can be filled with new frame data
and asynchronously started or the next output can be taken from the Infer Request and displayed.

This technique can be generalized to any available parallel slack, for example, doing inference and simultaneously
encoding the resulting (previous) frames or running further inference, like some emotion detection on top of
the face detection results.
There are important performance caveats though, for example the tasks that run in parallel should try to avoid
oversubscribing the shared compute resources.
As another example, if the inference is performed on the iGPU, and the CPU is essentially idle,
then it makes sense to do things on the CPU in parallel. But if the inference is performed say on the GPU,
then there is little gain from doing the (resulting video) encoding on the same GPU in parallel,
because the device is already busy.

This and other performance implications and tips for the Async API are covered in the
[Optimization Guide](https://docs.openvino.ai/2023.0/_docs_optimization_guide_dldt_optimization_guide.html).

Other demo objectives are:

* Video as input support via OpenCV\*
* Visualization of the resulting bounding boxes and text labels (from the `.labels` file)
  or class number (if no file is provided)

## How It Works

On startup, the application reads command-line parameters and loads a model to OpenVINO™ Runtime plugin. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

Async API operates with a notion of the "Infer Request" that encapsulates the inputs/outputs and separates
*scheduling and waiting for result*.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).

## Model API

The demo utilizes model wrappers, adapters and pipelines from [Python* Model API](../../common/python/model_zoo/model_api/README.md).

The generalized interface of wrappers with its unified results representation provides the support of multiple different object detection model topologies in one demo.

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/object_detection_demo/python/models.lst` file.
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

* architecture_type = centernet
  - ctdet_coco_dlav0_512
* architecture_type = ctpn
  - ctpn
* architecture_type = detr
  - detr-resnet50
* architecture_type = faceboxes
  - faceboxes-pytorch
* architecture_type = nanodet
  - nanodet-m-1.5x-416
* architecture_type = nanodet-plus
  - nanodet-plus-m-1.5x-416
* architecture_type = retinaface-pytorch
  - retinaface-resnet50-pytorch
* architecture_type = ssd
  - efficientdet-d0-tf
  - efficientdet-d1-tf
  - face-detection-0200
  - face-detection-0202
  - face-detection-0204
  - face-detection-0205
  - face-detection-0206
  - face-detection-adas-0001
  - face-detection-retail-0004
  - face-detection-retail-0005
  - faster-rcnn-resnet101-coco-sparse-60-0001
  - faster_rcnn_inception_resnet_v2_atrous_coco
  - faster_rcnn_resnet50_coco
  - pedestrian-and-vehicle-detector-adas-0001
  - pedestrian-detection-adas-0002
  - person-detection-0106
  - person-detection-0200
  - person-detection-0201
  - person-detection-0202
  - person-detection-0203
  - person-detection-0301
  - person-detection-0302
  - person-detection-0303
  - person-detection-retail-0013
  - person-vehicle-bike-detection-2000
  - person-vehicle-bike-detection-2001
  - person-vehicle-bike-detection-2002
  - person-vehicle-bike-detection-2003
  - person-vehicle-bike-detection-2004
  - product-detection-0001
  - retinanet-tf
  - rfcn-resnet101-coco-tf
  - ssd_mobilenet_v1_coco
  - ssd_mobilenet_v1_fpn_coco
  - ssd-resnet34-1200-onnx
  - ssdlite_mobilenet_v2
  - vehicle-detection-0200
  - vehicle-detection-0201
  - vehicle-detection-0202
  - vehicle-detection-adas-0002
  - vehicle-license-plate-detection-barrier-0106
  - vehicle-license-plate-detection-barrier-0123
* architecture_type = ultra_lightweight_face_detection
  - ultra-lightweight-face-detection-rfb-320
  - ultra-lightweight-face-detection-slim-320
* architecture_type = yolo
  - mobilenet-yolo-v4-syg
  - person-vehicle-bike-detection-crossroad-yolov3-1020
  - yolo-v1-tiny-tf
  - yolo-v2-ava-0001
  - yolo-v2-ava-sparse-35-0001
  - yolo-v2-ava-sparse-70-0001
  - yolo-v2-tf
  - yolo-v2-tiny-ava-0001
  - yolo-v2-tiny-ava-sparse-30-0001
  - yolo-v2-tiny-ava-sparse-60-0001
  - yolo-v2-tiny-tf
  - yolo-v2-tiny-vehicle-detection-0001
  - yolo-v3-tf
  - yolo-v3-tiny-tf
* architecture_type = yolov3-onnx
  - yolo-v3-onnx
  - yolo-v3-tiny-onnx
* architecture_type = yolov4
  - yolo-v4-tf
  - yolo-v4-tiny-tf
* architecture_type = yolof
  - yolof
* architecture_type = yolox
  - yolox-tiny

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: object_detection_demo.py [-h] -m MODEL -at
                                {centernet,detr,ctpn,faceboxes,nanodet,nanodet-plus,retinaface,retinaface-pytorch,ssd,ultra_lightweight_face_detection,yolo,yolov4,yolof,yolox,yolov3-onnx}
                                -i INPUT [--adapter {openvino,ovms}]
                                [-d DEVICE] [--labels LABELS] [-t PROB_THRESHOLD]
                                [--resize_type {standard,fit_to_window,fit_to_window_letterbox}]
                                [--input_size INPUT_SIZE INPUT_SIZE] [--anchors ANCHORS [ANCHORS ...]]
                                [--masks MASKS [MASKS ...]] [--layout LAYOUT]
                                [--num_classes NUM_CLASSES][-nireq NUM_INFER_REQUESTS] [-nstreams NUM_STREAMS]
                                [-nthreads NUM_THREADS] [--loop] [-o OUTPUT] [-limit OUTPUT_LIMIT] [--no_show]
                                [--output_resolution OUTPUT_RESOLUTION] [-u UTILIZATION_MONITORS]
                                [--reverse_input_channels] [--mean_values MEAN_VALUES MEAN_VALUES MEAN_VALUES]
                                [--scale_values SCALE_VALUES SCALE_VALUES SCALE_VALUES] [-r]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model or
                        address of model inference service if using OVMS adapter.
  -at, --architecture_type  Required. Specify model' architecture type. Valid values are {centernet,detr,ctpn,faceboxes,nanodet,nanodet-plus,retinaface,retinaface-pytorch,ssd,ultra_lightweight_face_detection,yolo,yolov4,yolof,yolox,yolov3-onnx}.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera id.
  --adapter {openvino,ovms}
                        Optional. Specify the model adapter. Default is
                        openvino.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU or
                        GPU is acceptable. The demo
                        will look for a suitable plugin for device specified.
                        Default value is CPU.

Common model options:
  --labels LABELS       Optional. Labels mapping file.
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections
                        filtering.
  --resize_type {standard,fit_to_window,fit_to_window_letterbox}
                        Optional. A resize type for model preprocess. By default used model predefined type.
  --input_size INPUT_SIZE INPUT_SIZE
                        Optional. The first image size used for CTPN model
                        reshaping. Default: 600 600. Note that submitted
                        images should have the same resolution, otherwise
                        predictions might be incorrect.
  --anchors ANCHORS [ANCHORS ...]
                        Optional. A space separated list of anchors. By default used default anchors for model. Only
                        for YOLOV4 architecture type.
  --masks MASKS [MASKS ...]
                        Optional. A space separated list of mask for anchors. By default used default masks for model.
                        Only for YOLOV4 architecture type.
  --layout LAYOUT       Optional. Model inputs layouts. Ex. NCHW or
                        input0:NCHW,input1:NC in case of more than one input.
  --num_classes NUM_CLASSES
                        Optional. Number of detected classes. Only for NanoDet, NanoDetPlus
                        architecture types.

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
                        Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086
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

Input transform options:
  --reverse_input_channels REVERSE_CHANNELS
                        Optional. Switch the input channels order from
                        BGR to RGB.
  --mean_values MEAN_VALUES
                        Optional. Normalize input by subtracting the mean
                        values per channel. Example: 255.0 255.0 255.0
  --scale_values SCALE_VALUES
                        Optional. Divide input by scale values per channel.
                        Division is applied after mean values subtraction.
                        Example: 255.0 255.0 255.0

Debug options:
  -r, --raw_output_message
                        Optional. Output inference results raw values showing.
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can use the following command to do inference on GPU with a pre-trained object detection model:

```sh
python3 object_detection_demo.py \
  -d GPU \
  -i <path_to_video>/inputVideo.mp4 \
  -m <path_to_model>/efficientdet-d0-tf.xml \
  -at ssd \
  --labels <omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt
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
  leads to a decrease in FPS. Please, keep this in mind and use the C++ version of this demo for performance-critical cases.

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
python3 object_detection_demo.py \
  -i <path_to_video>/inputVideo.mp4 \
  -m localhost:9000/models/object_detection \
  -at ssd \
  --labels <omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt \
  --adapter ovms
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
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
* [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
