# Security Barrier Camera С++ Demo

This demo showcases Vehicle and License Plate Detection network followed by the Vehicle Attributes Recognition and License Plate Recognition networks applied on top
of the detection results. You can use a set of the following pre-trained models with the demo:
* `vehicle-license-plate-detection-barrier-0106`, which is a primary detection network to find the vehicles and license plates
* `vehicle-attributes-recognition-barrier-0039`, which is executed on top of the results from the first network and
reports general vehicle attributes, for example, vehicle type (car/van/bus/track) and color
* `license-plate-recognition-barrier-0001`, which is executed on top of the results from the first network
and reports a string per recognized license plate

For more information about the pre-trained models, refer to the [model documentation](../../models/intel/index.md).

Other demo objectives are:
* Video/Camera as inputs, via OpenCV\*
* Example of complex asynchronous networks pipelining: Vehicle Attributes and License Plate Recognition networks are executed on top of the Vehicle Detection results
* Visualization of Vehicle Attributes and License Plate information for each detected object

## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. The Vehicle and License Plate
Detection network is required, the other two are optional.

The core component of the application pipeline is the Worker class, which executes incoming instances of a `Task` class.
`Task` is an abstract class that describes data to process and how to process the data.
For example, a `Task` can be to read a frame or to get detection results.
There is a pool of `Task` instances. These `Task`s are awaiting to be executed.
When a `Task` from the pool is being executed, it may create and/or submit another `Task` to the pool.
Each `Task` stores a smart pointer to an instance of `VideoFrame`, which represents an image the `Task` works with.
When the sequence of `Task`s is completed and none of the `Task`s require a `VideoFrame` instance, the `VideoFrame` is destroyed.
This triggers creation of a new sequence of `Task`s.
The pipeline of this demo executes the following sequence of `Task`s:
* `Reader`, which reads a new frame
* `InferTask`, which starts detection inference
* `RectExtractor`, which waits for detection inference to complete and runs a classifier and a recognizer
* `ResAggregator`, which draws the results of the inference on the frame
* `Drawer`, which shows the frame with the inference results

At the end of the sequence, the `VideoFrame` is destroyed and the sequence starts again for the next frame.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
[ INFO ] InferenceEngine:
    API version ............ <version>
    Build .................. <number>

interactive_vehicle_detection [OPTION]
Options:

    -h                         Print a usage message.
    -i "<path1>" "<path2>"     Required for video or image files input. Path to video or image files.
    -m "<path>"                Required. Path to the Vehicle and License Plate Detection model .xml file.
    -m_va "<path>"             Optional. Path to the Vehicle Attributes model .xml file.
    -m_lpr "<path>"            Optional. Path to the License Plate Recognition model .xml file.
      -l "<absolute_path>"     Required for CPU custom layers. Absolute path to a shared library with the kernels implementation.
          Or
      -c "<absolute_path>"     Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -d "<device>"              Optional. Specify the target device for Vehicle Detection (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_va "<device>"           Optional. Specify the target device for Vehicle Attributes (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_lpr "<device>"          Optional. Specify the target device for License Plate Recognition (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -pc                        Optional. Enables per-layer performance statistics.
    -r                         Optional. Output inference results as raw values.
    -t                         Optional. Probability threshold for vehicle and license plate detections.
    -no_show                   Optional. Do not show processed video.
    -auto_resize               Optional. Enable resizable input with support of ROI crop and auto resize.
    -nireq                     Optional. Number of infer requests. 0 sets the number of infer requests equal to the number of inputs.
    -nc                        Required for web camera input. Maximum number of processed camera inputs (web cameras).
    -fpga_device_ids           Optional. Specify FPGA device IDs (0,1,n).
    -loop_video                Optional. Enable playing video on a loop.
    -n_iqs                     Optional. Number of allocated frames. It is a multiplier of the number of inputs.
    -ni                        Optional. Specify the number of channels generated from provided inputs (with -i and -nc keys). For example, if only one camera is provided, but -ni is set to 2, the demo will process frames as if they are captured from two cameras. 0 sets the number of input channels equal to the number of provided inputs.
    -fps                       Optional. Set the playback speed not faster than the specified FPS. 0 removes the upper bound.
    -n_wt                      Optional. Set the number of threads including the main thread a Worker class will use.
    -display_resolution        Optional. Specify the maximum output window resolution.
    -tag                       Required for HDDL plugin only. If not set, the performance on Intel(R) Movidius(TM) X VPUs will not be optimal. Running each network on a set of Intel(R) Movidius(TM) X VPUs with a specific tag. You must specify the number of VPUs for each network in the hddl_service.config file. Refer to the corresponding README file for more information.
    -nstreams "<integer>"      Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)
    -nthreads "<integer>"      Optional. Number of threads to use for inference on the CPU (including HETERO and MULTI cases).

```

Running the application with an empty list of options yields an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to do inference on a GPU with the OpenVINO toolkit pre-trained models, run the following command:

```sh
./security_barrier_camera_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU
```

To do inference for two video inputs using two asynchronous infer request on FPGA with the OpenVINO toolkit pre-trained models, run the following command:
```sh
./security_barrier_camera_demo -i <path_to_video>/inputVideo_0.mp4 <path_to_video>/inputVideo_1.mp4 -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d HETERO:FPGA,CPU -d_va HETERO:FPGA,CPU -d_lpr HETERO:FPGA,CPU -nireq 2
```

> **NOTE**: For the `-tag` option (HDDL plugin only), you must specify the number of VPUs for each network in the `hddl_service.config` file located in the `<INSTALL_DIR>/deployment_tools/inference_engine/external/hddl/config/` direcrtory using the following tags:
> * `tagDetect` for the Vehicle and License Plate Detection network
> * `tagAttr` for the Vehicle Attributes Recognition network
> * `tagLPR` for the License Plate Recognition network
>
> For example, to run the sample on one Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Compact R card with eight Intel&reg; Movidius&trade; X VPUs:
> ```sh
> "service_settings":
> {
>  "graph_tag_map":{"tagDetect": 6, "tagAttr": 1, "tagLPR": 1}
> }
> ```


### Optimization Hints for Heterogeneous Scenarios with FPGA

If you build the Inference Engine with the OMP, you can use the following parameters for Heterogeneous scenarois:

* `OMP_NUM_THREADS`: Specifies number of threads to use. For heterogeneous scenarios with FPGA, when several inference requests are used asynchronously, limiting the number of CPU threads with `OMP_NUM_THREADS` allows to avoid competing for resources between threads. For the Security Barrier Camera Demo, recommended value is `OMP_NUM_THREADS=1`.
* `KMP_BLOCKTIME`: Sets the time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping. The default value is 200ms, which is not optimal for the demo. Recommended value is `KMP_BLOCKTIME=1`.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo has been tested on the following Model Downloader available topologies: 
>* `license-plate-recognition-barrier-0001`
>* `vehicle-attributes-recognition-barrier-0039`
>* `vehicle-license-plate-detection-barrier-0106`
> Other models may produce unexpected results on these devices.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
