# Object Detection SSD C++ Demo, Async API Performance Showcase

This demo showcases Object Detection with SSD and new Async API.
Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the app can continue doing things on the host, while accelerator is busy.
Specifically, this demo keeps the number of Infer Requests that you have set using `nireq` flag. While some of the Infer Requests are processed by IE, the other ones can be filled with new frame data and asynchronously started or the next output can be taken from the Infer Request and displayed.

> **NOTE:** This topic describes usage of C++ implementation of the Object Detection SSD Demo Async API. For the Python* implementation, refer to [Object Detection SSD Python* Demo, Async API Performance Showcase](../python_demos/object_detection_demo_ssd_async/README.md).

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
* Demonstration of the Async API in action, so the demo features two modes (toggled by the Tab key)
    - "User specified" mode, where you can set the number of Infer Requests, throughput streams and threads. Inference, starting new requests and displaying the results of completed requests are all performed asynchronously. The purpose of this mode is to get the higher FPS by fully utilizing all available devices.
    - "Min latency" mode, which uses only one Infer Request. The purpose of this mode is to get the lowest latency.

## How It Works

On the start-up, the application reads command line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture it performs inference and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

New "Async API" operates with new notion of the "Infer Request" that encapsulates the inputs/outputs and separates *scheduling and waiting for result*,
next section.

The pipeline is the same for both modes. The difference is in the number of Infer Requests used.
```cpp
    while (true) {
        if (Infer Request containing the next video frame has completed) {
            get inference results
            process inference results
            display the frame
        } else if (one of the Infer Requests is idle and it is not the end of the input video) {
            capture frame
            populate empty InferRequest
            set completion callback
            start InferRequest
        }
    }
```

### Async API

The Inference Engine also offers new API based on the notion of Infer Requests. One specific usability upside
is that the requests encapsulate the inputs and outputs allocation, so you just need to access the blob  with `GetBlob` method.

More importantly, you can execute a request asynchronously (in the background) and wait until ready, when the result is actually needed.
In a mean time your app can continue :

```cpp
// load network
InferenceEngine::Core ie;
auto network = ie.ReadNetwork("Model.xml");
// populate inputs etc
auto input = async_infer_request.GetBlob(input_name);
...
// start the async Infer Request (puts the request to the queue and immediately returns)
async_infer_request->StartAsync();
// here you can continue execution on the host until results of the current request are really needed
//...
async_infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);
auto output = async_infer_request.GetBlob(output_name);
```
Notice that there is no direct way to measure execution time of the Infer Request that is running asynchronously, unless
you measure the Wait executed immediately after the StartAsync. But this essentially would mean the serialization and synchronous
execution.
That is why the inference speed is not reported. It is recommended to use the [Benchmark App](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html) for measuring inference speed of different models.


For more details on the requests-based Inference Engine API, including the Async execution, refer to [Integrate the Inference Engine New Request API with Your Application](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Integrate_with_customer_application_new_API.html).


## Running

Running the application with the `-h` option yields the following usage message:
```
./object_detection_demo_ssd_async -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

object_detection_demo_ssd_async [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path>"               Required. Path to a video file (specify "cam" to work with camera).
    -m "<path>"               Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
          Or
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with the kernel descriptions.
    -d "<device>"             Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -labels "<path>"          Optional. Path to a file with labels mapping.
    -pc                       Optional. Enables per-layer performance report.
    -r                        Optional. Inference results as raw values.
    -t                        Optional. Probability threshold for detections.
    -auto_resize              Optional. Enables resizable input with support of ROI crop & auto resize.
    -nireq "<integer>"        Optional. Number of Infer Requests.
    -nthreads "<integer>"     Optional. Number of threads.
    -nstreams                 Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)
    -loop_input               Optional. Iterate over input infinitely.
    -no_show                  Optional. Do not show processed video.
    -u                        Optional. List of monitors to show initially.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

You can use the following command to do inference on GPU with a pre-trained object detection model:
```sh
./object_detection_demo_ssd_async -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/ssd.xml -d GPU
```

The only GUI knob is using **Tab** to switch between "User specified" mode and "Min latency" mode.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
The demo reports
* **FPS**: average rate of video frame processing (frames per second)
* **Latency**: average time required to process one frame (from reading the frame to displaying the results)
You can use both of these metrics to measure application-level performance.


## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
