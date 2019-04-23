# Security Barrier Camera С++ Demo

This demo showcases Vehicle and License Plate Detection network followed by the Vehicle Attributes Recognition and License Plate Recognition networks applied on top
of the detection results. You can use a set of the following pre-trained models with the demo:
* `vehicle-license-plate-detection-barrier-0106`, which is a primary detection network to find the vehicles and license plates
* `vehicle-attributes-recognition-barrier-0039`, which is executed on top of the results from the first network and
reports general vehicle attributes, for example, vehicle type (car/van/bus/track) and color
* `license-plate-recognition-barrier-0001`, which is executed on top of the results from the first network
and reports a string per recognized license plate

For more information about the pre-trained models, refer to the [Open Model Zoo](https://github.com/opencv/open_model_zoo/tree/master/intel_models/index.md) repository on GitHub*.

Other demo objectives are:
* Video/Camera as inputs, via OpenCV\*
* Example of simple asynchronous networks pipelining: Vehicle Attributes and License Plate Recognition networks are executed on top of the Vehicle Detection results
* Visualization of Vehicle Attributes and License Plate information for each detected object

## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. The Vehicle and License Plate
Detection network is required, the other two are optional.

Upon getting a frame from the OpenCV VideoCapture, the applications performs inference of Vehicles and License Plate Detection network, then performs
another two inferences of Vehicle Attributes and License Plate Recognition networks if they were specified in command line, and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Specify Input Shapes** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./security_barrier_camera_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

interactive_vehicle_detection [OPTION]
Options:

    -h                         Print a usage message.
    -i "<path1>" "<path2>"     Required. Path to video or image files. Default value is "cam" to work with cameras.
    -m "<path>"                Required. Path to the Vehicle and License Plate Detection model .xml file.
    -m_va "<path>"             Optional. Path to the Vehicle Attributes model .xml file.
    -m_lpr "<path>"            Optional. Path to the License Plate Recognition model .xml file.
      -l "<absolute_path>"     Optional. For CPU custom layers, if any. Absolute path to a shared library with the kernels implementation.
          Or
      -c "<absolute_path>"     Optional. For GPU custom kernels, if any. Absolute path to an .xml file with the kernels description.
    -d "<device>"              Optional. Specify the target device for Vehicle Detection (CPU, GPU, FPGA, MYRIAD, HDDL or HETERO).
    -d_va "<device>"           Optional. Specify the target device for Vehicle Attributes (CPU, GPU, FPGA, MYRIAD, HDDL or HETERO).
    -d_lpr "<device>"          Optional. Specify the target device for License Plate Recognition (CPU, GPU, FPGA, MYRIAD, HDDL or HETERO).
    -pc                        Optional. Enable per-layer performance statistics.
    -r                         Optional. Output inference results as raw values.
    -t                         Optional. Probability threshold for vehicle and license plate detections.
    -no_show                   Optional. Do not show processed video.
    -auto_resize               Optional. Enable resizable input with support of ROI crop and auto resize.
    -nireq                     Optional. Number of infer request for pipelined mode (default value is 1).
    -nc                        Optional. Number of processed cameras (default value is 1) if the input (-i) is specified as camera.
    -fpga_device_ids           Optional. Specify FPGA device IDs (0,1,n).
    -loop_video                Optional. Enable playing video on a loop.
    -ni                        Optional. Specify the number of inputs to be processed.
    -display_resolution        Optional. Specify the maximum output window resolution.
```

Running the application with an empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to do inference on a GPU with the OpenVINO toolkit pre-trained models, run the following command:

```sh
./security_barrier_camera_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU
```

To do inference for two video inputs using two asynchronous infer request on FPGA with the OpenVINO toolkit pre-trained models, run the following command:
```sh
./security_barrier_camera_demo -i <path_to_video>/inputVideo_0.mp4 <path_to_video>/inputVideo_1.mp4 -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d HETERO:FPGA,CPU -d_va HETERO:FPGA,CPU -d_lpr HETERO:FPGA,CPU -nireq 2
```

### Optimization Hints for Heterogeneous Scenarios with FPGA

* `OMP_NUM_THREADS`: Specifies number of threads to use. For heterogeneous scenarios with FPGA, when several inference requests are used asynchronously, limiting the number of CPU threads with `OMP_NUM_THREADS` allows to avoid competing for resources between threads. For the Security Barrier Camera Demo, recommended value is `OMP_NUM_THREADS=1`.
* `KMP_BLOCKTIME`: Sets the time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping. The default value is 200ms, which is not optimal for the demo. Recommended value is `KMP_BLOCKTIME=1`.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.


## See Also
* [Using Open Model Zoo demos](https://github.com/opencv/open_model_zoo/tree/master/demos/README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader)
