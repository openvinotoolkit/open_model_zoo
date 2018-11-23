Inference Engine Demos
================

The Inference Engine demo applications are simple console applications that demonstrate how you can use the Intel's Deep Learning Inference Engine in your applications.

The Deep Learning Inference Engine release package provides the following demo applications available in the demos
directory in the Inference Engine installation directory:

 - [Crossroad Camera Demo](./crossroad_camera_demo/README.md) - Person Detection followed by the Person Attributes Recognition and Person Reidentification Retail, supports images/video and camera inputs. *NEW MODELS SHOWCASE, below*.
 - [End to end video analytics end2end_video_analytics_ie Demo](./end2end_video_analytics/end2end_video_analytics_ie/README.md) - End to end demo application for image classification with inference engine.
 - [End to end video analytics end2end_video_analytics_opencv Demo](./end2end_video_analytics/end2end_video_analytics_opencv/README.md) - End to end demo application for image classification with OpenCVDNN.
 - [Human Pose Estimation Demo](./human_pose_estimation_demo/README.md) - Human pose estimation demo. *NEW MODELS SHOWCASE, below*.
 - [Image Segmentation Demo](./segmentation_demo/README.md) - Inference of image segmentation networks like FCN8 (the demo supports only images as inputs).
 - [Interactive Face Detection Demo](./interactive_face_detection_demo/README.md) - Face Detection coupled with Age/Gender, Head-Pose, Emotion, and Facial Landmarks detectors. Supports video and camera inputs.  *NEW MODELS SHOWCASE, below*.
 - [Mask R-CNN Demo for TensorFlow* Object Detection API models](./mask_rcnn_demo/README.md) - Inference of semantic segmentation networks created with TensorFlow\* Object Detection API.
 - [Multi-Channel Face Detection Demo](./multichannel_face_detection/README.md) - Simultaneous Multi Camera Face Detection demo.
 - [Object Detection Demo](./object_detection_demo/README.md) - Inference of object detection networks like Faster R-CNN (the demo supports only images as inputs).
 - [Object Detection for SSD Demo app](./object_detection_demo_ssd_async/README.md) - Demo application for SSD-based Object Detection networks, new Async API performance showcase, and simple OpenCV interoperability (supports video and camera inputs).
 - [Object Detection for YOLO V3 Demo app](./object_detection_demo_yolov3_async/README.md) - Demo application for YOLOV3-based Object Detection networks, new Async API performance showcase, and simple OpenCV interoperability (supports video and camera inputs).
 - [Pedestrian Tracker Demo](./pedestrian_tracker_demo/README.md) - Demo application for pedestrian tracking scenario.
 - [Security Barrier Camera Demo](./security_barrier_camera_demo/README.md) - Vehicle Detection followed by the Vehicle Attributes and License-Plate Recognition, supports images/video and camera inputs. *NEW MODELS SHOWCASE, below*.
 - [Smart Classroom Demo](./smart_classroom_demo/README.md) - Face recognition and action detection demo for classroom environment. *NEW MODELS SHOWCASE, below*.
 - [Super Resolution Demo](./super_resolution_demo/README.md) - Super Resolution demo (the demo supports only images as inputs).

*Few demos referenced above have simplified equivalents in Python (`python_demos` subfolder)*.

## Demos that Support Pre-Trained Models Shipped with the Product

(!) Important Note: Inference Engine MYRIAD and FPGA plugins are available in [proprietary](https://software.intel.com/en-us/openvino-toolkit) distribution only.

The product includes several pre-trained  [models] (../intel_models/index.html).
The table below shows the correlation between models and demos/plugins (_the plugins names are exactly as they are passed to the demos with `-d` option_).

| Model                                           | Demoss supported on the model                                                                     | CPU         | GPU         |HETERO:FPGA,CPU| MYRIAD    |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------|-------------|-------------|---------------|-----------|
|   face-detection-adas-0001                      | [Interactive Face Detection Demo](./interactive_face_detection_demo/README.md)    | Supported   | Supported   | Supported     | Supported |
|   age-gender-recognition-retail-0013            | [Interactive Face Detection Demo](./interactive_face_detection_demo/README.md)    | Supported   | Supported   | Supported     | Supported |
|   head-pose-estimation-adas-0001                | [Interactive Face Detection Demo](./interactive_face_detection_demo/README.md)    | Supported   | Supported   | Supported     | Supported |
|   emotions-recognition-retail-0003              | [Interactive Face Detection Demo](./interactive_face_detection_demo/README.md)    | Supported   | Supported   | Supported     | Supported |
|   facial-landmarks-35-adas-0001                 | [Interactive Face Detection Demo](./interactive_face_detection_demo/README.md)    | Supported   | Supported   | Supported     |           |
|   vehicle-license-plate-detection-barrier-0106  | [Security Barrier Camera Demo](./security_barrier_camera_demo/README.md)          | Supported   | Supported   | Supported     | Supported |
|   vehicle-attributes-recognition-barrier-0039   | [Security Barrier Camera Demo](./security_barrier_camera_demo/README.md)          | Supported   | Supported   | Supported     | Supported |
|   license-plate-recognition-barrier-0001        | [Security Barrier Camera Demo](./security_barrier_camera_demo/README.md)          | Supported   | Supported   | Supported     | Supported |
|   person-detection-retail-0001                  | [Object Detection Demo](./object_detection_demo/README.md)                       | Supported   | Supported   | Supported     |           |
|   person-vehicle-bike-detection-crossroad-0078  | [Crossroad Camera Demo](./crossroad_camera_demo/README.md)                       | Supported   | Supported   | Supported     | Supported |
|   person-attributes-recognition-crossroad-0031  | [Crossroad Camera Demo](./crossroad_camera_demo/README.md)                       | Supported   | Supported   | Supported     | Supported |
|   person-reidentification-retail-0031           | [Crossroad Camera Demo](./crossroad_camera_demo/README.md)<br>[Pedestrian Tracker Demo](./pedestrian_tracker_demo/README.md) | Supported   | Supported   | Supported     | Supported |
|   person-reidentification-retail-0076           | [Crossroad Camera Demo](./crossroad_camera_demo/README.md)                       | Supported   | Supported   | Supported     | Supported |
|   person-reidentification-retail-0079           | [Crossroad Camera Demo](./crossroad_camera_demo/README.md)                       | Supported   | Supported   | Supported     | Supported |
|   road-segmentation-adas-0001                   | [Image Segmentation Demo](./segmentation_demo/README.md)                        | Supported   | Supported   |               |           |
|   semantic-segmentation-adas-0001               | [Image Segmentation Demo](./segmentation_demo/README.md)                        | Supported   | Supported   |               |           |
|   person-detection-retail-0013                  | any demo that supports SSD\*-based models, above<br>[Pedestrian Tracker Demo](./pedestrian_tracker_demo/README.md) | Supported   | Supported   | Supported     | Supported |
|   face-detection-retail-0004                    | any demo that supports SSD\*-based models, above                                                   | Supported   | Supported   | Supported     | Supported |
|   face-person-detection-retail-0002             | any demo that supports SSD\*-based models, above                                                   | Supported   | Supported   | Supported     | Supported |
|   pedestrian-detection-adas-0002                | any demo that supports SSD\*-based models, above                                                   | Supported   | Supported   | Supported     |           |
|   vehicle-detection-adas-0002                   | any demo that supports SSD\*-based models, above                                                   | Supported   | Supported   | Supported     | Supported |
|   pedestrian-and-vehicle-detector-adas-0001     | any demo that supports SSD\*-based models, above                                                   | Supported   | Supported   | Supported     |           |
|   person-detection-action-recognition-0003      | [Smart Classroom Demo](./smart_classroom_demo/README.md)                         | Supported   | Supported   | Supported     |           |
|   landmarks-regression-retail-0009              | [Smart Classroom Demo](./smart_classroom_demo/README.md)                         | Supported   | Supported   | Supported     |           |
|   face-reidentification-retail-0071             | [Smart Classroom Demo](./smart_classroom_demo/README.md)                         | Supported   | Supported   | Supported     | Supported |
|   human-pose-estimation-0001                    | [Human Pose Estimation Demo](./human_pose_estimation_demo/README.md)          | Supported   | Supported   | Supported     |           |
|   single-image-super-resolution-0034            | [Super Resolution Demo](./super_resolution_demo/README.md)                     | Supported   |             |               |           |

*Few demos referenced above have simplified equivalents in Python (`python_demos` subfolder)*.

Notice that the FPGA support comes through a [heterogeneous execution](@ref PluginHETERO), for example, when the post-processing is happening on the CPU.

## Building the Demo Applications

To be able to build demos you need to source _InferenceEngine_ and _OpenCV_ environment from a binary package which is available as [proprietary](https://software.intel.com/en-us/openvino-toolkit) distribution.
Please run the following command (assuming that the binary package was installed to <INSTALL_DIR>):
```sh
source <INSTALL_DIR>/deployment_tools/bin/setupvars.sh
```
Also, you can build IE binaries from the _dldt_ repo. In this case please set `InferenceEngine_DIR` to a CMake folder you built the dldt binaries from.
Please also set the `OpenCV_DIR` variable pointing to the required OpenCV package.

### Linux* OS
The officially supported Linux build environment is the following:

* Ubuntu* 16.04 LTS 64-bit or CentOS* 7.4 64-bit
* GCC* 5.4.0 (for Ubuntu* 16.04) or GCC* 4.8.5 (for CentOS* 7.4)
* CMake* version 2.8 or higher.
* OpenCV 3.3 or later (required for some demos and demos)

<br>You can build the demo applications using the _CMake_ file in the `demos` directory.

Create a new directory and change your current directory to the new one:
```sh
mkdir build
cd build
```
Run _CMake_ to generate Make files:
```sh
cmake -DCMAKE_BUILD_TYPE=Release <path_to_inference_engine_demos_directory>
```

To build demos with debug information, use the following command:
```sh
cmake -DCMAKE_BUILD_TYPE=Debug <path_to_inference_engine_demos_directory>
```

Run _Make_ to build the application:
```sh
make
```

After that you can find binaries for all demos applications in the `intel64/Release` subfolder.

### Microsoft Windows* OS

The recommended Windows build environment is the following:
* Microsoft Windows* 10
* Microsoft* Visual Studio* 2015 including Microsoft Visual Studio 2015 Community or Microsoft Visual Studio 2017
* CMake* version 2.8 or later
* OpenCV* 3.3 or later


Generate Microsoft Visual Studio solution file using `create_msvc2015_solution.bat` file or `create_msvc2017_solution.bat` file
and then build the resulting solution `Demos.sln` in the Microsoft Visual Studio 2015 or Microsoft Visual Studio 2015 accordingly.

## Running the Demo Applications

Before running compiled binary files, make sure your application can find the Inference Engine libraries.
Use the `setupvars.sh` script (or `setupvars.bat` on Windows), which will set all necessary environment variables pointing to the binaries from the installed
binary package you installed to the `<INSTALL_DIR>`.

For that, run:
```
source <INSTALL_DIR>/deployment_tools/bin/setupvars.sh
```
on Linux or
```
source <INSTALL_DIR>/deployment_tools/bin/setupvars.bat
```
to source required environment on Windows.
<br>If you are using _Inference Engine_ binaries from the _dldt_ repository then you need to configure `LD_LIBRARY_PATH` variable (or `PATH` on Windows) manually.

What is left is running the required demo with appropriate commands, providing IR information (typically with "-m" command-line option).
Please note that Inference Engine assumes that weights are in the same folder as _.xml_ file.

---
\* Other names and brands may be claimed as the property of others.
