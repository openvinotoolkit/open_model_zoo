# Face Detection Using MTCNN Python\* Demo

This demo showcases Face Detection using MTCNN architecture.

The MTCNN architecture consists of three different models, which perform face detection in three serial stages - 
Proposal, Refine and Output.

This and other performance implications and tips for the Async API are covered in the
[Optimization Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html).

Other demo objectives are:
* Video as input support via OpenCV\*
* Visualization of the resulting bounding boxes, confidences and landmarks

## How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

The MTCNN model is perfomed in three serial stages:
- Proposal stage - the MTCNN-P model reshapes into several separate models depending on image size. Each model performes inference and gives a pack of "proposal" boxes
- Refine stage - each "proposal" box refines by MTCNN-R model execution. The model could be executed once for all "proposal" boxes (*by default*) which involve its reshaping by setting new batch size and reloading to device, or several times in asynchronous mode with fixed batch size (see `--refine_batch_size` option) 
- Output stage - each "refined" box corrects by MTCNN-O model execution. The model could be executed once for all "refined" boxes (*by default*) which involve its reshaping by setting new batch size and reloading to device, or several times in asynchronous mode with fixed batch size (see `--output_batch_size` option)

## Running

Running the application with the `-h` option yields the following usage message:
```
python3 object_detection_demo.py -h
```
The command yields the following usage message:
```
usage: face_detection_demo_mtcnn.py [-h] -i INPUT -m_p MODEL_PROPOSAL -m_r
                                    MODEL_REFINE -m_o MODEL_OUTPUT
                                    [-d_p DEVICE_PROPOSAL]
                                    [-d_r DEVICE_REFINE] [-d_o DEVICE_OUTPUT]
                                    [--proposal_async]
                                    [--refine_batch_size REFINE_BATCH_SIZE]
                                    [--refine_requests REFINE_REQUESTS]
                                    [--refine_nstreams REFINE_NSTREAMS]
                                    [--refine_nthreads REFINE_NTHREADS]
                                    [--output_batch_size OUTPUT_BATCH_SIZE]
                                    [--output_requests OUTPUT_REQUESTS]
                                    [--output_nstreams OUTPUT_NSTREAMS]
                                    [--output_nthreads OUTPUT_NTHREADS]
                                    [--loop] [-o OUTPUT] [-limit OUTPUT_LIMIT]
                                    [--no_show] [-u UTILIZATION_MONITORS] [-r]

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera
                        id.
  -m_p MODEL_PROPOSAL, --model_proposal MODEL_PROPOSAL
                        Required. Path to an .xml file with a trained model.
  -m_r MODEL_REFINE, --model_refine MODEL_REFINE
                        Required. Path to an .xml file with a trained model.
  -m_o MODEL_OUTPUT, --model_output MODEL_OUTPUT
                        Required. Path to an .xml file with a trained model.
  -d_p DEVICE_PROPOSAL, --device_proposal DEVICE_PROPOSAL
                        Optional. Specify the target device to infer on MTCNN
                        Proposal stage; CPU, GPU, FPGA, HDDL or MYRIAD is
                        acceptable. The sample will look for a suitable plugin
                        for device specified. Default value is CPU.
  -d_r DEVICE_REFINE, --device_refine DEVICE_REFINE
                        Optional. Specify the target device to infer on MTCNN
                        Refine stage; CPU, GPU, FPGA, HDDL or MYRIAD is
                        acceptable. The sample will look for a suitable plugin
                        for device specified. Default value is CPU.
  -d_o DEVICE_OUTPUT, --device_output DEVICE_OUTPUT
                        Optional. Specify the target device to infer on MTCNN
                        Output stage; CPU, GPU, FPGA, HDDL or MYRIAD is
                        acceptable. The sample will look for a suitable plugin
                        for device specified. Default value is CPU.

Inference options:
  --proposal_async
  --refine_batch_size REFINE_BATCH_SIZE
                        Optional. Sets fixed batch size for MTCNN Refine
                        stage, disallowing dynamic reshape.Forces asynchronous
                        mode for MTCNN Refine stage. Dynamic reshape is set by
                        default.
  --refine_requests REFINE_REQUESTS
                        Optional. Number of infer requests for MTCNN Refine
                        stage. Works only with --refine_batch_size > 0.
  --refine_nstreams REFINE_NSTREAMS
                        Optional. Number of streams to use for inference on
                        the CPU or/and GPU in throughput mode (for HETERO and
                        MULTI device cases use format <device1>:<nstreams1>,
                        <device2>:<nstreams2> or just <nstreams>) for MTCNN
                        Refine stage.Works only with --refine_batch_size > 0.
  --refine_nthreads REFINE_NTHREADS
                        Optional. Number of threads to use for inference on
                        CPU (including HETERO cases) forMTCNN Refine stage.
                        Works only with --refine_batch_size > 0.
  --output_batch_size OUTPUT_BATCH_SIZE
                        Optional. Sets fixed batch size for MTCNN Output
                        stage, disallowing dynamic reshape.Forces asynchronous
                        mode for MTCNN Output stage. Dynamic reshape is set by
                        default.
  --output_requests OUTPUT_REQUESTS
                        Optional. Number of infer requests for MTCNN Output
                        stage. Works only with --output_batch_size > 0.
  --output_nstreams OUTPUT_NSTREAMS
                        Optional. Number of streams to use for inference on
                        the CPU or/and GPU in throughput mode (for HETERO and
                        MULTI device cases use format <device1>:<nstreams1>,
                        <device2>:<nstreams2> or just <nstreams>) for MTCNN
                        Output stage.Works only with --output_batch_size > 0.
  --output_nthreads OUTPUT_NTHREADS
                        Optional. Number of threads to use for inference on
                        CPU (including HETERO cases) forMTCNN Output stage.
                        Works only with --output_batch_size > 0.

Input/output options:
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output. If -1
                        is set, all frames are stored.
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
You can use the following command to do inference on CPU with a pre-trained model:
```
python3 face_detection_demo_mtcnn.py -i <path_to_video>/inputVideo.mp4 -m_p <path_to_model>/mtcnn-p.xml -m_r <path_to_model>/mtcnn-r.xml -m_o <path_to_model>/mtcnn-o.xml
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
