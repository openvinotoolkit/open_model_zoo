Inference Engine Demos
================

The Inference Engine demo applications are simple console applications that demonstrate how you can use the Intel's Deep Learning Inference Engine in your applications.

The Deep Learning Inference Engine release package provides the following demo applications available in the demos
directory in the Inference Engine installation directory:

 - [CPU Extensions](@ref CPUExtensions) library with topology-specific layers (like DetectionOutput used in the SSD*, below)
 - [Crossroad Camera Demo](@ref InferenceEngineCrossroadCameraDemoApplication) - Person Detection followed by the Person Attributes Recognition and Person Reidentification Retail, supports images/video and camera inputs. *NEW MODELS SHOWCASE, below*.
 - [End to end video analytics end2end_video_analytics_ie Demo](@ref InferenceEngine_end2end_video_analytics_end2end_video_analytics_ie_DemoApplication) - End to end demo application for image classification with inference engine.
 - [End to end video analytics end2end_video_analytics_opencv Demo](@ref InferenceEngine_end2end_video_analytics_end2end_video_analytics_opencv_DemoApplication) - End to end demo application for image classification with OpenCVDNN.
 - [Image Segmentation Demo](@ref InferenceEngineSegmentationDemoApplication) - Inference of image segmentation networks like FCN8 (the sample supports only images as inputs)
 - [Interactive Face Detection Demo](@ref InferenceEngineInteractiveFaceDetectionDemoApplication) - Face Detection coupled with Age-Gender and Head-Pose, supports video and camera inputs.  *NEW MODELS SHOWCASE, below*.
 - [Mask R-CNN Demo for TensorFlow* Object Detection API models](@ref InferenceEngineMaskRCNNDemoApplication) - Inference of semantic segmentation networks created with TensorFlow* Object Detection API.
 - [Multi-Channel Face Detection Demo](@ref InferenceEngineMultiChannelFaceDetectionDemoApplication) - Simultaneous Multi Camera Face Detection demo.
 - [Object Detection Demo](@ref InferenceEngineObjectDetectionDemoApplication) - Inference of object detection networks like Faster R-CNN (the sample supports only images as inputs)
 - [Object Detection for SSD Demo app](@ref InferenceEngineObjectDetectionSSDDemoAsyncApplication) - Demo application for SSD-based Object Detection networks, new Async API performance showcase, and simple OpenCV interoperability (supports video and camera inputs)
 - [Security Barrier Camera Demo](@ref InferenceEngineSecurityBarrierCameraDemoApplication) - Vehicle Detection followed by the Vehicle Attributes and License-Plate Recognition, supports images/video and camera inputs. *NEW MODELS SHOWCASE, below*.
 - [Smart Classroom Demo](@ref InferenceEngineSmartClassroomDemoApplication) - Face recognition and action detection demo for classroom environment. *NEW MODELS SHOWCASE, below*

*Few demos referenced above have simplified equivalents in Python (<code>python_demos</code> subfolder)*.

## Demos that Support Pre-Trained Models Shipped with the Product

(!) Important Note: Inference Engine MYRIAD and FPGA plugins are available in [propiretary](https://software.intel.com/en-us/openvino-toolkit) distribution only.

The product includes several pre-trained  [models] (../../intel_models/index.html).
The table below shows the correlation between models and demos/plugins (_the plugins names are exactly as they are passed to the demos with `-d` option_). The correlation between the plugins and supported devices see in the [Supported Devices](@ref SupportedPlugins) section. The demos are available in `<INSTALL_DIR>/deployment_tools/inference_engine/samples`.

| Model                                           | Demoss supported on the model                                                                     | CPU         | GPU         |HETERO:FPGA,CPU| MYRIAD    |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------|-------------|-------------|---------------|-----------|
|   face-detection-adas-0001                      | [Interactive Face Detection Demo](@ref InferenceEngineInteractiveFaceDetectionSampleApplication)  | Supported   | Supported   | Supported     | Supported |
|   age-gender-recognition-retail-0013            | [Interactive Face Detection Demo](@ref InferenceEngineInteractiveFaceDetectionDemoApplication)    | Supported   | Supported   | Supported     | Supported |
|   head-pose-estimation-adas-0001                | [Interactive Face Detection Demo](@ref InferenceEngineInteractiveFaceDetectionDemoApplication)    | Supported   | Supported   | Supported     | Supported |
|   emotions-recognition-retail-0003              | [Interactive Face Detection Demo](@ref InferenceEngineInteractiveFaceDetectionDemoApplication)    | Supported   | Supported   | Supported     |           |
|   vehicle-license-plate-detection-barrier-0106  | [Security Barrier Camera Demo](@ref InferenceEngineSecurityBarrierCameraDemoApplication)          | Supported   | Supported   |               | Supported |
|   vehicle-attributes-recognition-barrier-0039   | [Security Barrier Camera Demo](@ref InferenceEngineSecurityBarrierCameraDemoApplication)          | Supported   | Supported   | Supported     | Supported |
|   license-plate-recognition-barrier-0001        | [Security Barrier Camera Demo](@ref InferenceEngineSecurityBarrierCameraDemoApplication)          | Supported   | Supported   |               | Supported |
|   person-detection-retail-0001                  | [Object Detection Demo](@ref InferenceEngineObjectDetectionDemoApplication)                       | Supported   | Supported   |               |           |
|   person-vehicle-bike-detection-crossroad-0078  | [Crossroad Camera Demo](@ref InferenceEngineCrossroadCameraDemoApplication)                       | Supported   | Supported   |               | Supported |
|   person-attributes-recognition-crossroad-0031  | [Crossroad Camera Demo](@ref InferenceEngineCrossroadCameraDemoApplication)                       | Supported   | Supported   | Supported     |           |
|   person-reidentification-retail-0031           | [Crossroad Camera Demo](@ref InferenceEngineCrossroadCameraDemoApplication)                       | Supported   | Supported   |               | Supported |
|   person-reidentification-retail-0079           | [Crossroad Camera Demo](@ref InferenceEngineCrossroadCameraDemoApplication)                       | Supported   | Supported   |               |           |
|   road-segmentation-adas-0001                   | [Image Segmentation Demo](@ref InferenceEngineSegmentationDemoApplication)                        | Supported   | Supported   |               |           |
|   semantic-segmentation-adas-0001               | [Image Segmentation Demo](@ref InferenceEngineSegmentationDemoApplication)                        | Supported   | Supported   |               |           |
|   person-detection-retail-0013                  | any Demo that supports SSD*-based models, above                                                   | Supported   | Supported   | Supported     | Supported |
|   face-detection-retail-0004                    | any Demo that supports SSD*-based models, above                                                   | Supported   | Supported   | Supported     | Supported |
|   face-person-detection-retail-0002             | any Demo that supports SSD*-based models, above                                                   | Supported   | Supported   |               | Supported |
|   pedestrian-detection-adas-0002                | any Demo that supports SSD*-based models, above                                                   | Supported   | Supported   | Supported     | Supported |
|   vehicle-detection-adas-0002                   | any Demo that supports SSD*-based models, above                                                   | Supported   | Supported   | Supported     | Supported |
|   pedestrian-and-vehicle-detector-adas-0001     | any Demo that supports SSD*-based models, above                                                   | Supported   | Supported   |               | Supported |
|   person-detection-action-recognition-0001      | [Smart classroom Demo](@ref InferenceEngineSmartClassroomDemoApplication)                         | Supported   | Supported   |               | Supported |
|   landmarks-regression-retail-0001              | [Smart classroom Demo](@ref InferenceEngineSmartClassroomDemoApplication)                         | Supported   | Supported   |               |           |
|   face-reidentification-retail-0001             | [Smart classroom Demo](@ref InferenceEngineSmartClassroomDemoApplication)                         | Supported   | Supported   |               | Supported |

*Few demos referenced above have simplified equivalents in Python (<code>python_demos</code> subfolder)*.

Notice that the FPGA support comes through a [heterogeneous execution](@ref PluginHETERO), for example, when the post-processing is happening on the CPU.

## <a name="build_demos_linux"></a> Building the Demo Applications on Linux*
The officially supported Linux build environment is the following:

* Ubuntu* 16.04 LTS 64-bit or CentOS* 7.4 64-bit
* GCC* 5.4.0 (for Ubuntu* 16.04) or GCC* 4.8.5 (for CentOS* 7.4)
* CMake* version 2.8 or higher.
* OpenCV 3.3 or later (required for some demos and demos)

<br>You can build the demo applications using the <i>CMake</i> file in the `demos` directory.

Create a new directory and change your current directory to the new one:
```sh
mkdir build
cd build
```
Run <i>CMake</i> to generate Make files:
```sh
cmake -DCMAKE_BUILD_TYPE=Release <path_to_inference_engine_demos_directory>
```

To build demos with debug information, use the following command:
```sh
cmake -DCMAKE_BUILD_TYPE=Debug <path_to_inference_engine_demos_directory>
```

Run <i>Make</i> to build the application:
```sh
make
```

For ease of reference, the Inference Engine installation folder is referred to as <code><INSTALL_DIR></code>.

After that you can find binaries for all demos applications in the <code>intel64/Release</code> subfolder.

## <a name="build_demos_windows"></a> Building the Demo Applications on Microsoft Windows* OS

The recommended Windows build environment is the following:
* Microsoft Windows* 10
* Microsoft* Visual Studio* 2015 including Microsoft Visual Studio 2015 Community or Microsoft Visual Studio 2017
* CMake* version 2.8 or later
* OpenCV* 3.3 or later


Generate Microsoft Visual Studio solution file using <code>create_msvc_solution.bat</code> file in the <code>demos</code> directory and then build the solution <code>demos\build\demos.sln</code> in the Microsoft Visual Studio 2015.

## Running the Sample Applications

Before running compiled binary files, make sure your application can find the Inference Engine libraries.
Use the `setvars.sh` script, which will set all necessary environment variables.

For that, run (assuming that you are in a <code><INSTALL_DIR>/deployment_tools/inference_engine/bin/intel64/Release</code> folder):
<pre>
source ../../setvars.sh
</pre>

What is left is running the required demo with appropriate commands, providing IR information (typically with "-m" command-line option).
Please note that Inference Engine assumes that weights are in the same folder as _.xml_ file.

---
\* Other names and brands may be claimed as the property of others.