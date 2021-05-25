# Security Barrier Camera C++ Demo

![example](./security_barrier_camera.gif)

This demo showcases Vehicle and License Plate Detection network followed by the Vehicle Attributes Recognition and License Plate Recognition networks applied on top
of the detection results.

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

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/security_barrier_camera_demo/cpp/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported Models

* license-plate-recognition-barrier-0001
* license-plate-recognition-barrier-0007
* vehicle-attributes-recognition-barrier-0039
* vehicle-attributes-recognition-barrier-0042
* vehicle-license-plate-detection-barrier-0106
* vehicle-license-plate-detection-barrier-0123

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
[ INFO ] InferenceEngine: <version>

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
    -no_show                   Optional. Don't show output.
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
    -u                         Optional. List of monitors to show initially.
```

Running the application with an empty list of options yields an error message.

For example, to do inference on a GPU with the OpenVINO toolkit pre-trained models, run the following command:

```sh
./security_barrier_camera_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU
```

To do inference for two video inputs using two asynchronous infer request, run the following command:

```sh
./security_barrier_camera_demo -i <path_to_video>/inputVideo_0.mp4 <path_to_video>/inputVideo_1.mp4 -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d CPU -nireq 2
```

To do inference for video inputs on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, some optimization hints are suggested to make good use of the computation ability:

* configuring the number of allocated frames (`-n_iqs`) to provide enough inputs for inference;
* configuring the number of infer request (`-nireq`) to achieve asynchronous inference;
* configuring the number of threads (`-n_wt`) for multi-threaded processing.

For example, to run the sample on one Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Compact R card, run the following command:

```sh
./security_barrier_camera_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d HDDL -d_va HDDL -d_lpr HDDL -n_iqs 10 -n_wt 4 -nireq 10
```

> **NOTE**: For the `-tag` option (HDDL plugin only), you must specify the number of VPUs for each network in the `hddl_service.config` file located in the `<INSTALL_DIR>/deployment_tools/inference_engine/external/hddl/config/` directory using the following tags:
> * `tagDetect` for the Vehicle and License Plate Detection network
> * `tagAttr` for the Vehicle Attributes Recognition network
> * `tagLPR` for the License Plate Recognition network
>
> For example, to run the sample on one Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Compact R card with eight Intel&reg; Movidius&trade; X VPUs:
> ```json
> "service_settings":
> {
>  "graph_tag_map":{"tagDetect": 6, "tagAttr": 1, "tagLPR": 1}
> }
> ```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
