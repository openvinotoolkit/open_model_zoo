# Social Distance C++ Demo

This reference implementation showcases a retail social distance application that detects people and measures the distance between them. If this distance is less than a value previously provided by the user, then an alert is triggered. This application uses the [person-detection-retail-0013](https://docs.openvinotoolkit.org/2021.3/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) and [person-reidentification-retail-0300](https://docs.openvinotoolkit.org/2021.3/_models_intel_person_reidentification_retail_0300_description_person_reidentification_retail_0300.html) Intel® pre-trained models, that can be downloaded using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that is used by the application.

To download the models, run the following command:

```bash
mkdir models
cd models
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 --precisions FP16-INT8
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-reidentification-retail-0300 --precisions FP16-INT8
```

For more information about the pre-trained models, refer to the [model documentation](../../models/intel/index.md).

Other demo objectives are:
* Video/Camera as inputs, via OpenCV\*
* Example of complex asynchronous networks pipelining: Person Re-Identification network is executed on top of the Person Detection results
* Visualization of the minimum social distancing threshold violation

## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. Both Person Detection and Re-Identification networks are required.

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
* `DetectionsProcessor`, which waits for detection inference to complete and runs a Re-Identification model
* `ResAggregator`, which draws the results of the inference on the frame
* `Drawer`, which shows the frame with the inference results

At the end of the sequence, the `VideoFrame` is destroyed and the sequence starts again for the next frame.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```
[ INFO ] InferenceEngine: <version>

interactive_vehicle_detection [OPTION]
Options:

    -h                         Print a usage message.
    -i "<path1>" "<path2>"     Required for video or image files input. Path to video or image files.
    -m_det "<path>"            Required. Path to the Person Detection model .xml file.
    -m_reid "<path>"           Optional. Path to the Person Re-Identification model .xml file.
      -l "<absolute_path>"     Required for CPU custom layers. Absolute path to a shared library with the kernels implementation.
          Or
      -c "<absolute_path>"     Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -d_det "<device>"          Optional. Specify the target device for Person Detection (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_reid "<device>"         Optional. Specify the target device for Person Re-Identification (the list of available devices is shown below). Default value is CPU. Use "-d_reid HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -pc                        Optional. Enables per-layer performance statistics.
    -r                         Optional. Output inference results as raw values.
    -t                         Optional. Probability threshold for person detections.
    -no_show                   Optional. Do not show processed video.
    -auto_resize               Optional. Enable resizable input with support of ROI crop and auto resize.
    -nireq                     Optional. Number of infer requests. 0 sets the number of infer requests equal to the number of inputs.
    -nc                        Required for web camera input. Maximum number of processed camera inputs (web cameras).
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

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to do inference on a GPU with the OpenVINO toolkit pre-trained models, run the following command:

```sh
./social_distance_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/person-detection-retail-0013.xml -m_reid <path_to_model>/person-reidentification-retail-0300.xml -d_det GPU
```

To do inference for two video inputs using two asynchronous infer request on CPU with the OpenVINO toolkit pre-trained models, run the following command:
```sh
./social_distance_demo -i <path_to_video>/inputVideo_0.mp4 <path_to_video>/inputVideo_1.mp4 -m <path_to_model>/person-detection-retail-0013.xml -m_reid <path_to_model>/person-reidentification-retail-0300.xml -d_det CPU -d_reid CPU -nireq 2
```

To do inference for video inputs on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, some optimization hints are suggested to make good use of the computation ability:

* configuring the number of allocated frames (`-n_iqs`) to provide enough inputs for inference;
* configuring the number of infer request (`-nireq`) to achieve asynchronous inference;
* configuring the number of threads (`-n_wt`) for multi-threaded processing.

For example, to run the sample on one Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Compact R card, run the following command:
```sh
./social_distance_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/person-detection-retail-0013.xml -m_reid <path_to_model>/person-reidentification-retail-0300.xml  -d_det HDDL -d_reid HDDL -n_iqs 10 -n_wt 4 -nireq 10
```

> **NOTE**: For the `-tag` option (HDDL plugin only), you must specify the number of VPUs for each network in the `hddl_service.config` file located in the `<INSTALL_DIR>/deployment_tools/inference_engine/external/hddl/config/` directory using the following tags:
> * `tagDetect` for the Person Detection network
> * `tagReId` for the Person Re-Identification network
>
> For example, to run the sample on one Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Compact R card with eight Intel&reg; Movidius&trade; X VPUs:
> ```json
> "service_settings":
> {
>  "graph_tag_map":{"tagDetect": 6, "tagReId": 1}
> }
> ```


## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo has been tested on the following Model Downloader available topologies:
>* `person-detection-retail-0013`
>* `person-reidentification-retail-0300`
> Other models may produce unexpected results on these devices.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
