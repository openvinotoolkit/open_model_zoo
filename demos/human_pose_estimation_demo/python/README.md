# Human Pose Estimation Python\* Demo

![](../human_pose_estimation.gif)

This demo showcases the work of multi-person 2D pose estimation algorithms. The task is to predict a pose: body skeleton, which consists of a predefined set of keypoints and connections between them, for every person in an input image/video.

Demo application supports inference in both sync and async modes. Please refer to [Optimization Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html) for more information about Async API and its use.

Other demo objectives are:
* Video as input support via OpenCV\*
* Visualization of the resulting poses
* Demonstration of the Async API in action. For this, the demo features two modes toggled by the **Tab** key:
    - "User specified" mode, where you can set the number of Infer Requests, throughput streams and threads.
      Inference, starting new requests and displaying the results of completed requests are all performed asynchronously.
      The purpose of this mode is to get the higher FPS by fully utilizing all available devices.
    - "Min latency" mode, which uses only one Infer Request. The purpose of this mode is to get the lowest latency.

## How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work
with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your
model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about
the argument, refer to **When to Reverse Input Channels** section of
[Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```
python3 human_pose_estimation.py -h
```
The command yields the following usage message:
```
usage: human_pose_estimation.py [-h] -i INPUT [--loop] [-o OUTPUT]
                                -m MODEL -at {ae,openpose}
                                [--tsize TSIZE] [-t PROB_THRESHOLD] [-r]
                                [-d DEVICE] [-nireq NUM_INFER_REQUESTS]
                                [-nstreams NUM_STREAMS]
                                [-nthreads NUM_THREADS] [-loop LOOP]
                                [-no_show] [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera id.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output to save.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -at {ae,openpose}, --architecture_type {ae,openpose}
                        Required. Type of the network, either "ae" for
                        Associative Embedding or "openpose" for OpenPose.
  --tsize TSIZE         Optional. Target input size. This demo implements
                        image pre-processing pipeline that is common to human
                        pose estimation approaches. Image is resize first to
                        some target size and then the network is reshaped to
                        fit the input image shape. By default target image
                        size is determined based on the input shape from IR.
                        Alternatively it can be manually set via this
                        parameter. Note that for OpenPose-like nets image is
                        resized to a predefined height, which is the target
                        size in this case. For Associative Embedding-like nets
                        target size is the length of a short image side.
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for poses filtering.
  -r, --raw_output_message
                        Optional. Output inference results raw values showing.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU.
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests.
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        Optional. Number of streams to use for inference on
                        the CPU or/and GPU in throughput mode (for HETERO and
                        MULTI device cases use format
                        <device1>:<nstreams1>,<device2>:<nstreams2> or just
                        <nstreams>).
  -nthreads NUM_THREADS, --num_threads NUM_THREADS
                        Optional. Number of threads to use for inference on
                        CPU (including HETERO cases).
  -no_show, --no_show   Optional. Don't show output.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Running the application with the empty list of options yields the short usage message and an error message.
You can use the following command to do inference on CPU with a pre-trained human pose estimation model:
```
python3 human_pose_estimation.py -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/hpe.xml -d CPU
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

The demo uses OpenCV to display the resulting frame with estimated poses.
The demo reports
* **FPS**: average rate of video frame processing (frames per second)
* **Latency**: average time required to process one frame (from reading the frame to displaying the results)
You can use both of these metrics to measure application-level performance.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
