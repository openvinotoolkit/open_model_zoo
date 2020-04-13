# Object Detection SSD C++ Demo, Async API Performance Showcase

This demo showcases Object Detection with SSD and new Async API.
Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the app can continue doing things on the host, while accelerator is busy.
Specifically, this demo keeps two parallel infer requests and while the current is processed, the input frame for the next
is being captured. This essentially hides the latency of capturing, so that the overall framerate is rather
determined by the `MAXIMUM(detection time, input capturing time)` and not the `SUM(detection time, input capturing time)`.

> **NOTE:** This topic describes usage of C++ implementation of the Object Detection SSD Demo Async API. For the Python* implementation, refer to [Object Detection SSD Python* Demo, Async API Performance Showcase](../python_demos/object_detection_demo_ssd_async/README.md).

The technique can be generalized to any available parallel slack, for example, doing inference and simultaneously encoding the resulting
(previous) frames or running further inference, like some emotion detection on top of the face detection results.
There are important performance
caveats though, for example the tasks that run in parallel should try to avoid oversubscribing the shared compute resources.
For example, if the inference is performed on the FPGA, and the CPU is essentially idle, than it makes sense to do things on the CPU
in parallel. But if the inference is performed say on the GPU, than it can take little gain to do the (resulting video) encoding
on the same GPU in parallel, because the device is already busy.

This and other performance implications and tips for the Async API are covered in the [Optimization Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html)

Other demo objectives are:
* Video as input support via OpenCV
* Visualization of the resulting bounding boxes and text labels (from the .labels file) or class number (if no file is provided)
* OpenCV is used to draw resulting bounding boxes, labels, so you can copy paste this code without
need to pull Inference Engine demos helpers to your app
* Demonstration of the Async API in action, so the demo features two modes (toggled by the Tab key)
    -  Old-style "Sync" way, where the frame capturing with OpenCV executes back to back with the Detection
    -  "Truly Async" way when the Detection performed on the current frame, while the OpenCV captures the next one.

## How It Works

On the start-up, the application reads command line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture it performs inference and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

New "Async API" operates with new notion of the "Infer Request" that encapsulates the inputs/outputs and separates *scheduling and waiting for result*,
next section. And here what makes the performance look different:
* In the default ("Sync") mode the frame is captured and then immediately processed, below in pseudo-code:
```cpp
    while(true) {
        capture frame
        populate CURRENT InferRequest
        start CURRENT InferRequest //this call is async and returns immediately
        wait for the CURRENT InferRequest
        display CURRENT result
    }
```
So, this is rather reference implementation, where the new Async API is used in the serialized/synch fashion.
* In the "true" ASync mode the frame is captured and then immediately processed:
```cpp
    while(true) {
            capture frame
            populate NEXT InferRequest
            start NEXT InferRequest //this call is async and returns immediately
                wait for the CURRENT InferRequest (processed in a dedicated thread)
                display CURRENT result
            swap CURRENT and NEXT InferRequests
        }
```
In this case, the NEXT request is populated in the main (app) thread, while the CURRENT request is processed
(this is handled in the dedicated thread, internal to the IE runtime).

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
// start the async infer request (puts the request to the queue and immediately returns)
async_infer_request->StartAsync();
// here you can continue execution on the host until results of the current request are really needed
//...
async_infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);
auto output = async_infer_request.GetBlob(output_name);
```
Notice that there is no direct way to measure execution time of the infer request that is running asynchronously, unless
you measure the Wait executed immediately after the StartAsync. But this essentially would mean the serialization and synchronous
execution. This is what demo does for the default "SYNC" mode and reports as the "Detection time/fps" message on the screen.
In the truly asynchronous ("ASYNC") mode the host continues execution in the master thread, in parallel to the infer request.
And if the request is completed earlier than the Wait is called in the main thread (i.e. earlier than OpenCV decoded a new frame),
that reporting the time between StartAsync and Wait would obviously incorrect.
That is why in the "ASYNC" mode the inference speed is not reported.


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
    -pc                       Optional. Enables per-layer performance report.
    -r                        Optional. Inference results as raw values.
    -t                        Optional. Probability threshold for detections.
    -auto_resize              Optional. Enables resizable input with support of ROI crop & auto resize.
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

The only GUI knob is using **Tab** to switch between the synchronized execution and the true Async mode.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode the demo reports
* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and displaying the results.
* **Detection time**: inference time for the (object detection) network. It is reported in the "SYNC" mode only.
* **Wallclock time**, which is combined (application level) performance.


## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
