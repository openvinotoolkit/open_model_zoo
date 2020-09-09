# Object Detection SSD Python* Demo, Async API performance showcase

This demo showcases Object Detection with SSD and Async API.

Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the app can continue doing things on the host, while accelerator is busy.
Specifically, this demo keeps the number of Infer Requests that you have set using `-nireq` flag.
While some of the Infer Requests are processed by IE, the other ones can be filled with new frame data
and asynchronously started or the next output can be taken from the Infer Request and displayed.

The technique can be generalized to any available parallel slack, for example, doing inference and simultaneously
encoding the resulting (previous) frames or running further inference, like some emotion detection on top of
the face detection results.
There are important performance caveats though, for example the tasks that run in parallel should try to avoid
oversubscribing the shared compute resources.
For example, if the inference is performed on the FPGA, and the CPU is essentially idle,
than it makes sense to do things on the CPU in parallel. But if the inference is performed say on the GPU,
than it can take little gain to do the (resulting video) encoding on the same GPU in parallel,
because the device is already busy.

This and other performance implications and tips for the Async API are covered in the
[Optimization Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html).

Other demo objectives are:
* Video as input support via OpenCV\*
* Visualization of the resulting bounding boxes and text labels (from the `.labels` file)
  or class number (if no file is provided)
* Demonstration of the Async API in action. For this, the demo features two modes toggled by the **Tab** key:
    - "User specified" mode, where you can set the number of Infer Requests, throughput streams and threads.
      Inference, starting new requests and displaying the results of completed requests are all performed asynchronously.
      The purpose of this mode is to get the higher FPS by fully utilizing all available devices.
    - "Min latency" mode, which uses only one Infer Request. The purpose of this mode is to get the lowest latency.

## How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

Async API operates with a notion of the "Infer Request" that encapsulates the inputs/outputs and separates
*scheduling and waiting for result*.

The pipeline is the same for both modes. The difference is in the number of Infer Requests used.
```
while True:
    if (Infer Request containing the next video frame has completed):
        get inference results
        process inference results
        display the frame
    elif (one of the Infer Requests is idle and it is not the end of the input video):
        capture frame
        populate empty Infer Request
        start Infer Request
```

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work
with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your
model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about
the argument, refer to **When to Reverse Input Channels** section of
[Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

### Async API

The Inference Engine offers Async API based on the notion of Infer Requests. You can execute a Infer Requests
asynchronously (in the background) and wait until ready, when the result is actually needed.
In a mean time your app can continue :

```python
# load network as usual
ie = IECore()
net = ie.read_network(model='Model.xml', weights='Model.bin')
# load network to the plugin, setting the maximal number of concurrent Infer Requests to be used
exec_net = ie.load_network(network=net, device_name='GPU', num_requests=2)
# start concurrent Infer Requests (put requests to the queue and immediately return)
for i, request in enumerate(exec_net.requests):
    request.async_infer(inputs={'data': imgs[i]})
# here you can continue execution on the host until results of requests are really needed
# ...
outputs = [request.wait(-1) for request in exec_net.requests]
```

Another option is to set a callback on Infer Request completion:

```python
# load network as usual
ie = IECore()
net = ie.read_network(model='Model.xml', weights='Model.bin')
# load network to the plugin, setting the maximal number of concurrent Infer Requests to be used
exec_net = ie.load_network(network=net, device_name='GPU', num_requests=2)
# define a callback function
def callback(status, py_data):
    request, id = py_data
    print(id, {key: blob.buffer for key, blob in request.output_blobs.items()})

# start concurrent Infer Requests and set their completion callbacks
for i, request in enumerate(exec_net.requests):
    request.set_completion_callback(py_callback=callback, py_data=(request, i))
    request.async_infer(inputs={'data': imgs[i]})
    
# here you can continue execution on the host until results of requests are really needed
# ...
```

For more details on the requests-based Inference Engine API, including the Async execution, refer to
[Integrate the Inference Engine New Request API with Your Application](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Integrate_with_customer_application_new_API.html).


## Running

Running the application with the `-h` option yields the following usage message:
```
python3 object_detection_demo_ssd_async.py -h
```
The command yields the following usage message:
```
usage: object_detection_demo_ssd_async.py [-h] -m MODEL -i INPUT [-d DEVICE]
                                          [--labels LABELS]
                                          [-t PROB_THRESHOLD] [-r]
                                          [-nireq NUM_INFER_REQUESTS]
                                          [-nstreams NUM_STREAMS]
                                          [-nthreads NUM_THREADS]
                                          [-loop_input LOOP_INPUT] [-no_show]
                                          [-u UTILIZATION_MONITORS]
                                          [--keep_aspect_ratio]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to an image, video file or a numeric
                        camera ID.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU.
Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to an image, video file or a numeric
                        camera ID.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU.
  --labels LABELS       Optional. Labels mapping file.
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections
                        filtering.
  -r, --raw_output_message
                        Optional. Output inference results raw values showing.
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        Optional. Number of streams to use for inference on
                        the CPU or/and GPU in throughput mode (for HETERO and
                        MULTI device cases use format
                        <device1>:<nstreams1>,<device2>:<nstreams2> or just
                        <nstreams>)
  -nthreads NUM_THREADS, --num_threads NUM_THREADS
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to an image, video file or a numeric
                        camera ID.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU.
  --labels LABELS       Optional. Labels mapping file.
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections
                        filtering.
  -r, --raw_output_message
                        Optional. Output inference results raw values showing.
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        Optional. Number of streams to use for inference on
                        the CPU or/and GPU in throughput mode (for HETERO and
                        MULTI device cases use format
                        <device1>:<nstreams1>,<device2>:<nstreams2> or just
                        <nstreams>)
  -nthreads NUM_THREADS, --num_threads NUM_THREADS
                        Optional. Number of threads to use for inference on
                        CPU (including HETERO cases)
  -loop_input LOOP_INPUT, --loop_input LOOP_INPUT
                        Optional. Number of times to repeat the input.
  -no_show, --no_show   Optional. Don't show output
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
  --keep_aspect_ratio   Optional. Keeps aspect ratio on resize.
```

The number of Infer Requests is specified by `-nireq` flag. An increase of this number usually leads to an increase
of performance (throughput), since in this case several Infer Requests can be processed simultaneously if the device
supports parallelization. However, a large number of Infer Requests increases the latency because each frame still
has to wait before being sent for inference.

For higher FPS, it is recommended that you set `-nireq` to slightly exceed the `-nstreams` value,
summed across all devices used.

> **NOTE**: This demo is based on the callback functionality from the Inference Engine Python API.
  The selected approach makes the execution in multi-device mode optimal by preventing wait delays caused by
  the differences in device performance. However, the internal organization of the callback mechanism in Python API
  leads to FPS decrease. Please, keep it in mind and use the C++ version of this demo for performance-critical cases.

Running the application with the empty list of options yields the usage message given above and an error message.
You can use the following command to do inference on GPU with a pre-trained object detection model:
```
python3 object_detection_demo_ssd_async.py -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/ssd.xml -d GPU
```

To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO
[Model Downloader](../../../tools/downloader/README.md) or from
[https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine
format (\*.xml + \*.bin) using the
[Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

The only GUI knob is to use **Tab** to switch between the synchronized execution ("Min latency" mode)
and the asynchronous mode configured with provided command-line parameters ("User specified" mode).

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
The demo reports
* **FPS**: average rate of video frame processing (frames per second)
* **Latency**: average time required to process one frame (from reading the frame to displaying the results)
You can use both of these metrics to measure application-level performance.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
