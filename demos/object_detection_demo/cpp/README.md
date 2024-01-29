# Object Detection C++ Demo

![example](../object_detection.gif)

This demo showcases inference of Object Detection networks using Async API.
Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the app can continue doing things on the host, while accelerator is busy.
Specifically, this demo keeps the number of Infer Requests that you have set using `nireq` flag. While some of the Infer Requests are processed by OpenVINO™ Runtime, the other ones can be filled with new frame data and asynchronously started or the next output can be taken from the Infer Request and displayed.

This technique can be generalized to any available parallel slack, for example, doing inference and simultaneously encoding the resulting
(previous) frames or running further inference, like some emotion detection on top of the face detection results.
There are important performance caveats though, for example the tasks that run in parallel should try to avoid oversubscribing the shared compute resources.
For example, if the inference is performed on the iGPU, and the CPU is essentially idle, than it makes sense to do things on the CPU
in parallel. But if the inference is performed, say on the GPU, than it can take little gain to do the (resulting video) encoding
on the same GPU in parallel, because the device is already busy.

This and other performance implications and tips for the Async API are covered in the [Optimization Guide](https://docs.openvino.ai/2023.0/_docs_optimization_guide_dldt_optimization_guide.html)

Other demo objectives are:

* Video as input support via OpenCV
* Visualization of the resulting bounding boxes and text labels (from the labels file, see `-labels` option) or class number (if no file is provided)
* OpenCV is used to draw resulting bounding boxes, labels
* Demonstration of the Async API in action
* Demonstration of multiple models architectures support (including pre- and postprocessing) in one application

## How It Works

On startup, the application reads command-line parameters and loads a model to OpenVINO™ Runtime plugin. Upon getting a frame from the OpenCV VideoCapture it performs inference and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).

This demo operates in asynchronous manner by using "Infer Requests" that encapsulate the inputs/outputs and separates *scheduling and waiting for result*,
as shown in code mockup below:

```cpp
    while (true) {
        capture frame
        take empty InferRequest from pool
        if(empty InferRequest available) {
            populate empty InferRequest
            set completion callback
            submit InferRequest
        }

        while (there're completed InferRequests) {
            get inference results from InferRequest
            process inference results
            display the frame
        }
    }
```

For more details on the requests-based OpenVINO™ Runtime API, including the Async execution, refer to [Integrate the OpenVINO™ Runtime with Your Application](https://docs.openvino.ai/2023.0/openvino_docs_Integrate_OV_with_your_application.html).

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/object_detection_demo/cpp/models.lst` file.
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
* architecture_type = faceboxes
  - faceboxes-pytorch
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
  - retinaface-resnet50-pytorch
  - retinanet-tf
  - rfcn-resnet101-coco-tf
  - ssd-resnet34-1200-onnx
  - ssd_mobilenet_v1_coco
  - ssd_mobilenet_v1_fpn_coco
  - ssdlite_mobilenet_v2
  - vehicle-detection-0200
  - vehicle-detection-0201
  - vehicle-detection-0202
  - vehicle-detection-adas-0002
  - vehicle-license-plate-detection-barrier-0106
  - vehicle-license-plate-detection-barrier-0123
* architecture_type = yolo
  - mobilenet-yolo-v4-syg
  - person-vehicle-bike-detection-crossroad-yolov3-1020
  - yolo-v3-tf
  - yolo-v3-tiny-tf
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
  - yolo-v4-tf
  - yolo-v4-tiny-tf
  - yolof
* architecture_type = yolov3-onnx
  - yolo-v3-onnx
  - yolo-v3-tiny-onnx
* architecture_type = yolox
  - yolox-tiny

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the demo with `-h` shows this help message:
```
object_detection_demo [OPTION]
Options:

    -h                        Print a usage message.
    -at "<type>"              Required. Architecture type: centernet, faceboxes, retinaface, retinaface-pytorch, ssd, yolo, yolov3-onnx or yolox
    -i                        Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -m "<path>"               Required. Path to an .xml file with a trained model.
    -o "<path>"               Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086
    -limit "<num>"            Optional. Number of frames to store in output. If 0 is set, all frames are stored.
    -d "<device>"             Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -labels "<path>"          Optional. Path to a file with labels mapping.
    -layout "<string>"        Optional. Specify inputs layouts. Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.
    -r                        Optional. Inference results as raw values.
    -t                        Optional. Probability threshold for detections.
    -iou_t                    Optional. Filtering intersection over union threshold for overlapping boxes.
    -auto_resize              Optional. Enables resizable input with support of ROI crop & auto resize.
    -nireq "<integer>"        Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.
    -nthreads "<integer>"     Optional. Number of threads.
    -nstreams                 Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)
    -loop                     Optional. Enable reading the input in a loop.
    -no_show                  Optional. Don't show output.
    -output_resolution        Optional. Specify the maximum output window resolution in (width x height) format. Example: 1280x720. Input frame size used by default.
    -u                        Optional. List of monitors to show initially.
    -yolo_af                  Optional. Use advanced postprocessing/filtering algorithm for YOLO.
    -anchors                  Optional. A comma separated list of anchors. By default used default anchors for model. Only for YOLOV4 architecture type.
    -masks                    Optional. A comma separated list of mask for anchors. By default used default masks for model. Only for YOLOV4 architecture type.
    -reverse_input_channels   Optional. Switch the input channels order from BGR to RGB.
    -mean_values              Optional. Normalize input by subtracting the mean values per channel. Example: "255.0 255.0 255.0"
    -scale_values             Optional. Divide input by scale values per channel. Division is applied after mean values subtraction. Example: "255.0 255.0 255.0"
```

If labels file is used, it should correspond to model output. Demo treat labels, listed in the file, to be indexed from 0, one line - one label (that is very first line contains label for ID 0). Note that some models may return labels IDs in range 1..N, in this case label file should contain "background" label at the very first line.

You can use the following command to do inference on GPU with a pre-trained object detection model:

```sh
./object_detection_demo \
  -d GPU \
  -i <path_to_video>/inputVideo.mp4 \
  -m <path_to_model>/efficientdet-d0-tf.xml \
  -at ssd \
  -labels <omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt
```

>**NOTE**: If you provide a single image as an input, the demo processes and renders it quickly, then exits. To continuously visualize inference results on the screen, apply the `loop` option, which enforces processing a single image in a loop.

You can save processed results to a Motion JPEG AVI file or separate JPEG or PNG files using the `-o` option:

* To save processed results in an AVI file, specify the name of the output file with `avi` extension, for example: `-o output.avi`.
* To save processed results as images, specify the template name of the output image file with `jpg` or `png` extension, for example: `-o output_%03d.jpg`. The actual file names are constructed from the template at runtime by replacing regular expression `%03d` with the frame number, resulting in the following: `output_000.jpg`, `output_001.jpg`, and so on.
To avoid disk space overrun in case of continuous input stream, like camera, you can limit the amount of data stored in the output file(s) with the `limit` option. The default value is 1000. To change it, you can apply the `-limit N` option, where `N` is the number of frames to store.

>**NOTE**: Windows\* systems may not have the Motion JPEG codec installed by default. If this is the case, you can download OpenCV FFMPEG back end using the PowerShell script provided with the OpenVINO &trade; install package and located at `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. The script should be run with administrative privileges if OpenVINO &trade; is installed in a system protected folder (this is a typical case). Alternatively, you can save results as images.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
The demo reports:

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
