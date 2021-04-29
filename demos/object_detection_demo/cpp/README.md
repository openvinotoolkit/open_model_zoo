# Object Detection C++ Demo

![](../object_detection.gif)

This demo showcases Object Detection and Async API.
Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the app can continue doing things on the host, while accelerator is busy.
Specifically, this demo keeps the number of Infer Requests that you have set using `nireq` flag. While some of the Infer Requests are processed by IE, the other ones can be filled with new frame data and asynchronously started or the next output can be taken from the Infer Request and displayed.

> **NOTE:** This topic describes usage of C++ implementation of the Object Detection Demo Async API.

The technique can be generalized to any available parallel slack, for example, doing inference and simultaneously encoding the resulting
(previous) frames or running further inference, like some emotion detection on top of the face detection results.
There are important performance caveats though, for example the tasks that run in parallel should try to avoid oversubscribing the shared compute resources.
For example, if the inference is performed on the FPGA, and the CPU is essentially idle, than it makes sense to do things on the CPU
in parallel. But if the inference is performed say on the GPU, than it can take little gain to do the (resulting video) encoding
on the same GPU in parallel, because the device is already busy.

This and other performance implications and tips for the Async API are covered in the [Optimization Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html)

Other demo objectives are:

* Video as input support via OpenCV
* Visualization of the resulting bounding boxes and text labels (from the labels file, see `-labels` option) or class number (if no file is provided)
* OpenCV is used to draw resulting bounding boxes, labels, so you can copy paste this code without
need to pull Inference Engine demos helpers to your app
* Demonstration of the Async API in action
* Demonstration of multiple models architectures support (including pre- and postprocessing) in one application

## How It Works

On the start-up, the application reads command line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture it performs inference and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

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

For more details on the requests-based Inference Engine API, including the Async execution, refer to [Integrate the Inference Engine New Request API with Your Application](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Integrate_with_customer_application_new_API.html).


## Running

Running the application with the `-h` option yields the following usage message:
```
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

object_detection_demo [OPTION]
Options:

    -h                        Print a usage message.
    -at "<type>"              Required. Architecture type: centernet, faceboxes, retinaface, ssd or yolo
    -i                        Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -m "<path>"               Required. Path to an .xml file with a trained model.
    -o "<path>"               Optional. Name of output to save.
    -limit "<num>"            Optional. Number of frames to store in output. If 0 is set, all frames are stored.
      -l "<absolute_path>"    Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
          Or
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with the kernel descriptions.
    -d "<device>"             Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -labels "<path>"          Optional. Path to a file with labels mapping.
    -pc                       Optional. Enables per-layer performance report.
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
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in `<omz_dir>/demos/object_detection_demo/cpp/models.lst`.

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

If labels file is used, it should correspond to model output. Demo suggests labels listed in the file to be indexed from 0, one line - one label (i.e. very first line contains label for ID 0). Note that some models may return labels IDs in range 1..N, in this case label file should contain "background" label at the very first line.

You can use the following command to do inference on GPU with a pre-trained object detection model:
```sh
./object_detection_demo -i <path_to_video>/inputVideo.mp4 -at ssd -m <path_to_model>/ssd.xml -d GPU
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
The demo reports:

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).
You can use both of these metrics to measure application-level performance.


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
