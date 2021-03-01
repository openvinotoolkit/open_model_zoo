# Object Detection Python\* Demo

![](../object_detection.gif)

This demo showcases Object Detection with Sync and Async API.

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

## How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

Async API operates with a notion of the "Infer Request" that encapsulates the inputs/outputs and separates
*scheduling and waiting for result*.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work
with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your
model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about
the argument, refer to **When to Reverse Input Channels** section of
[Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```
python3 object_detection_demo.py -h
```
The command yields the following usage message:
```
usage: object_detection_demo.py [-h] -m MODEL -at
                                {ssd,yolo,faceboxes,centernet,retinaface} -i INPUT
                                [-d DEVICE] [--labels LABELS]
                                [-t PROB_THRESHOLD] [--keep_aspect_ratio]
                                [-nireq NUM_INFER_REQUESTS]
                                [-nstreams NUM_STREAMS] [-nthreads NUM_THREADS]
                                [--loop] [-o OUTPUT] [-limit OUTPUT_LIMIT] [--no_show]
                                [-u UTILIZATION_MONITORS] [-r]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -at {ssd,yolo,yolov4,faceboxes,centernet,ctpn,retinaface}, --architecture_type {ssd,yolo,yolov4,faceboxes,centernet,ctpn,retinaface}
                        Required. Specify model' architecture type.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera id.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU.

Common model options:
  --labels LABELS       Optional. Labels mapping file.
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections
                        filtering.
  --keep_aspect_ratio   Optional. Keeps aspect ratio on resize.
  --input_size          Optional. The first image size used for CTPN model reshaping.
                        Default: 600 600. Note that submitted images should have the same resolution,
                        otherwise predictions might be incorrect.

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
                        Optional. Name of output to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  --no_show             Optional. Don't show output.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.

Debug options:
  -r, --raw_output_message
                        Optional. Output inference results raw values showing.
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
[Model Downloader](../../../tools/downloader/README.md).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine
format (\*.xml + \*.bin) using the
[Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

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
