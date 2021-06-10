# Open Model Zoo Demos

The Open Model Zoo demo applications are console applications that provide robust application templates to help you implement specific deep learning scenarios. These applications involve increasingly complex processing pipelines that gather analysis data from several models that run inference simultaneously, such as detecting a person in a video stream along with detecting the person's physical attributes, such as age, gender, and emotional state

For the Intel® Distribution of OpenVINO™ toolkit, the demos are available after installation in the following directory: `<INSTALL_DIR>/deployment_tools/open_model_zoo/demos`.
The demos can also be obtained from the Open Model Zoo [GitHub repository](https://github.com/openvinotoolkit/open_model_zoo/).
C++, C++ G-API and Python\* versions are located in the `cpp`, `cpp_gapi` and `python` subdirectories respectively.

The Open Model Zoo includes the following demos:

- [3D Human Pose Estimation Python\* Demo](./human_pose_estimation_3d_demo/python/README.md) - 3D human pose estimation demo.
- [3D Segmentation Python\* Demo](./3d_segmentation_demo/python/README.md) - Segmentation demo segments 3D images using 3D convolutional networks.
- [Action Recognition Python\* Demo](./action_recognition_demo/python/README.md) - Demo application for Action Recognition algorithm, which classifies actions that are being performed on input video.
- [BERT Named Entity Recognition Python\* Demo](./bert_named_entity_recognition_demo/python/README.md) - NER Demo application that uses a CONLL2003-tuned BERT model for inference.
- [BERT Question Answering Python\* Demo](./bert_question_answering_demo/python/README.md)
- [BERT Question Answering Embedding Python\* Demo](./bert_question_answering_embedding_demo/python/README.md) - The demo demonstrates how to run BERT based models for question answering task.
- [Classification C++ Demo](./classification_demo/cpp/README.md) - Shows an example of using neural networks for image classification.
- [Colorization Python\* Demo](./colorization_demo/python/README.md) - Colorization demo colorizes input frames.
- [Crossroad Camera C++ Demo](./crossroad_camera_demo/cpp/README.md) - Person Detection followed by the Person Attributes Recognition and Person Reidentification Retail, supports images/video and camera inputs.
- [Deblurring Python\* Demo](./deblurring_demo/python/README.md) - Demo for deblurring the input images.
- [Face Detection MTCNN Python\* Demo](./face_detection_mtcnn_demo/python/README.md) - The demo demonstrates how to run MTCNN face detection model to detect faces on images.
- [Face Recognition Python\* Demo](./face_recognition_demo/python/README.md) - The interactive face recognition demo.
- [Formula Recognition Python\* Demo](./formula_recognition_demo/python/README.md) - The demo demonstrates how to run Im2latex formula recognition models and recognize latex formulas.
- [Gaze Estimation C++ Demo](./gaze_estimation_demo/cpp/README.md) - Face detection followed by gaze estimation, head pose estimation and facial landmarks regression.
- [Gaze Estimation C++ G-API Demo](./gaze_estimation_demo/cpp_gapi/README.md) - Face detection followed by gaze estimation, head pose estimation and facial landmarks regression. G-API version.
- [Gesture Recognition Python\* Demo](./gesture_recognition_demo/python/README.md) - Demo application for Gesture Recognition algorithm (e.g. American Sign Language gestures), which classifies gesture actions that are being performed on input video.
- [Handwritten Text Recognition Python\* Demo](./handwritten_text_recognition_demo/python/README.md) - The demo demonstrates how to run Handwritten Japanese Recognition models and Handwritten Simplified Chinese Recognition models.
- [Human Pose Estimation C++ Demo](./human_pose_estimation_demo/cpp/README.md) - Human pose estimation demo.
- [Human Pose Estimation Python\* Demo](./human_pose_estimation_demo/python/README.md) - Human pose estimation demo.
- [Image Inpainting Python\* Demo](./image_inpainting_demo/python/README.md) - Demo application for GMCNN inpainting network.
- [Image Processing C++ Demo](./image_processing_demo/cpp/README.md) - Demo application for deblurring and enhancing the resolution of the input image.
- [Image Retrieval Python\* Demo](./image_retrieval_demo/python/README.md) - The demo demonstrates how to run Image Retrieval models using OpenVINO&trade;.
- [Image Segmentation C++ Demo](./segmentation_demo/cpp/README.md) - Inference of semantic segmentation networks (supports video and camera inputs).
- [Image Segmentation Python\* Demo](./segmentation_demo/python/README.md) - Inference of semantic segmentation networks (supports video and camera inputs).
- [Image Translation Python\* Demo](./image_translation_demo/python/README.md) - Demo application to synthesize a photo-realistic image based on exemplar image.
- [Instance Segmentation Python\* Demo](./instance_segmentation_demo/python/README.md) - Inference of instance segmentation networks trained in `Detectron` or `maskrcnn-benchmark`.
- [Interactive Face Detection C++ Demo](./interactive_face_detection_demo/cpp/README.md) - Face Detection coupled with Age/Gender, Head-Pose, Emotion, and Facial Landmarks detectors. Supports video and camera inputs.
- [Interactive Face Detection G-API Demo](./interactive_face_detection_demo/cpp_gapi/README.md) - G-API based Face Detection coupled with Age/Gender, Head-Pose, Emotion, and Facial Landmarks detectors. Supports video and camera inputs.
- [Machine Translation Python\* Demo](./machine_translation_demo/python/README.md) - The demo demonstrates how to run non-autoregressive machine translation models.
- [Mask R-CNN C++ Demo for TensorFlow\* Object Detection API](./mask_rcnn_demo/cpp/README.md) - Inference of instance segmentation networks created with TensorFlow\* Object Detection API.
- [Monodepth Python\* Demo](./monodepth_demo/python/README.md) - The demo demonstrates how to run monocular depth estimation models.
- [Multi-Camera Multi-Target Tracking Python\* Demo](./multi_camera_multi_target_tracking_demo/python/README.md) Demo application for multiple targets (persons or vehicles) tracking on multiple cameras.
- [Multi-Channel Face Detection C++ Demo](./multi_channel_face_detection_demo/cpp/README.md) - The demo demonstrates an inference pipeline for multi-channel face detection scenario.
- [Multi-Channel Human Pose Estimation C++ Demo](./multi_channel_human_pose_estimation_demo/cpp/README.md) - The demo demonstrates an inference pipeline for multi-channel human pose estimation scenario.
- [Multi-Channel Object Detection Yolov3 C++ Demo](./multi_channel_object_detection_demo_yolov3/cpp/README.md) - The demo demonstrates an inference pipeline for multi-channel common object detection scenario.
- [Noise Suppression Python\* Demo](./noise_suppression_demo/python/README.md) - The demo shows how to use the OpenVINO™ toolkit to reduce noise in speech audio.
- [Object Detection Python\* Demo](./object_detection_demo/python/README.md) - Demo application for several object detection model types (like SSD, Yolo, etc).
- [Object Detection C++ Demo](./object_detection_demo/cpp/README.md) - Demo application for Object Detection networks (different models architectures are supported), async API showcase, simple OpenCV interoperability (supports video and camera inputs).
- [Pedestrian Tracker C++ Demo](./pedestrian_tracker_demo/cpp/README.md) - Demo application for pedestrian tracking scenario.
- [Place Recognition Python\* Demo](./place_recognition_demo/python/README.md) - This demo demonstrates how to run Place Recognition models using OpenVINO™.
- [Security Barrier Camera C++ Demo](./security_barrier_camera_demo/cpp/README.md) - Vehicle Detection followed by the Vehicle Attributes and License-Plate Recognition, supports images/video and camera inputs.
- [Speech Recognition DeepSpeech Python\* Demo](./speech_recognition_deepspeech_demo/python/README.md) - Speech recognition demo: accepts an audio file with an English phrase on input and converts it into text. This demo does streaming audio data processing and can optionally provide current transcription of the processed part.
- [Speech Recognition QuartzNet Python\* Demo](./speech_recognition_quartznet_demo/python/README.md) - Speech recognition demo for QuartzNet: takes a whole audio file with an English phrase on input and converts it into text.
- [Single Human Pose Estimation Python\* Demo](./single_human_pose_estimation_demo/python/README.md) - 2D human pose estimation demo.
- [Smart Classroom C++ Demo](./smart_classroom_demo/cpp/README.md) - Face recognition and action detection demo for classroom environment.
- [Social Distance C++ Demo](./social_distance_demo/cpp/README.md) - This demo showcases a retail social distance application that detects people and measures the distance between them.
- [Sound Classification Python\* Demo](./sound_classification_demo/python/README.md) - Demo application for sound classification algorithm.
- [Text Detection C++ Demo](./text_detection_demo/cpp/README.md) - Text Detection demo. It detects and recognizes multi-oriented scene text on an input image and puts a bounding box around detected area.
- [Text Spotting Python\* Demo](./text_spotting_demo/python/README.md) - The demo demonstrates how to run Text Spotting models.
- [Text-to-speech Python\* Demo](./text_to_speech_demo/python/README.md) - Shows an example of using Forward Tacotron and WaveRNN neural networks for text to speech task.
- [Time Series Forecasting Python\* Demo](./time_series_forecasting_demo/python/README.md) - The demo shows how to use the OpenVINO™ toolkit to time series forecastig.
- [Whiteboard Inpainting Python\* Demo](./whiteboard_inpainting_demo/python/README.md) - The demo shows how to use the OpenVINO™ toolkit to detect and hide a person on a video so that all text on a whiteboard is visible.

## Media Files Available for Demos

To run the demo applications, you can use images and videos from the media files collection available at https://github.com/intel-iot-devkit/sample-videos.

## Demos that Support Pre-Trained Models

> **NOTE:** Inference Engine HDDL plugin is available in [proprietary](https://software.intel.com/en-us/openvino-toolkit) distribution only.

You can download the [Intel pre-trained models](../models/intel/index.md) or [public pre-trained models](../models/public/index.md) using the OpenVINO [Model Downloader](../tools/downloader/README.md).

## Build the Demo Applications

To be able to build demos you need to source Inference Engine and OpenCV environment from a binary package which is available as [proprietary](https://software.intel.com/en-us/openvino-toolkit) distribution.
Please run the following command before the demos build (assuming that the binary package was installed to `<INSTALL_DIR>`):

```sh
source <INSTALL_DIR>/deployment_tools/bin/setupvars.sh
```

You can also build demos manually using Inference Engine built from the [openvino](https://github.com/openvinotoolkit/openvino) repo. In this case please set `InferenceEngine_DIR` environment variable to a folder containing `InferenceEngineConfig.cmake` and `ngraph_DIR` to a folder containing `ngraphConfig.cmake` in a build folder. Please also set the `OpenCV_DIR` to point to the OpenCV package to use. The same OpenCV version should be used both for Inference Engine and demos build. Alternatively these values can be provided via command line while running `cmake`. See [CMake's search procedure](https://cmake.org/cmake/help/latest/command/find_package.html#search-procedure).
Please refer to the Inference Engine [build instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode)
for details. Please also add path to built Inference Engine libraries to `LD_LIBRARY_PATH` (Linux*) or `PATH` (Windows*) variable before building the demos.

### <a name="build_demos_linux"></a>Build the Demo Applications on Linux*

The officially supported Linux* build environment is the following:

- Ubuntu* 18.04 LTS 64-bit or CentOS* 7.6 64-bit
- GCC* 7.5.0 (for Ubuntu* 18.04) or GCC* 4.8.5 (for CentOS* 7.6)
- CMake* version 3.10 or higher.

To build the demo applications for Linux, go to the directory with the `build_demos.sh` script and
run it:

```sh
build_demos.sh
```

You can also build the demo applications manually:

1. Navigate to a directory that you have write access to and create a demos build directory. This example uses a directory named `build`:
```sh
mkdir build
```
2. Go to the created directory:
```sh
cd build
```

3. Run CMake to generate the Make files for release or debug configuration:
  - For release configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Release <open_model_zoo>/demos
  ```
  - For debug configuration:
  ```sh
  cmake -DCMAKE_BUILD_TYPE=Debug <open_model_zoo>/demos
  ```
4. Run `cmake --build` tool to build the demos:
```sh
cmake --build .
```

For the release configuration, the demo application binaries are in `<path_to_build_directory>/intel64/Release/`;
for the debug configuration — in `<path_to_build_directory>/intel64/Debug/`.

### <a name="build_demos_windows"></a>Build the Demos Applications on Microsoft Windows* OS

The recommended Windows* build environment is the following:

- Microsoft Windows* 10
- Microsoft Visual Studio* 2017, or 2019
- CMake* version 3.10 or higher

> **NOTE**: If you want to use Microsoft Visual Studio 2019, you are required to install CMake 3.14.

To build the demo applications for Windows, go to the directory with the `build_demos_msvc.bat`
batch file and run it:

```bat
build_demos_msvc.bat
```

By default, the script automatically detects the highest Microsoft Visual Studio version installed on the machine and uses it to create and build
a solution for a demo code. Optionally, you can also specify the preferred Microsoft Visual Studio version to be used by the script. Supported
versions are: `VS2017`, `VS2019`. For example, to build the demos using the Microsoft Visual Studio 2017, use the following command:

```bat
build_demos_msvc.bat VS2017
```

The demo applications binaries are in the `C:\Users\<username>\Documents\Intel\OpenVINO\omz_demos_build\intel64\Release` directory.

You can also build a generated solution by yourself, for example, if you want to
build binaries in Debug configuration. Run the appropriate version of the
Microsoft Visual Studio and open the generated solution file from the `C:\Users\<username>\Documents\Intel\OpenVINO\omz_demos_build\Demos.sln`
directory.

You can also build the demo applications using `cmake --build` tool:
1. Navigate to a directory that you have write access to and create a demos build directory. This example uses a directory named `build`:
```
md build
```
2. Go to the created directory:
```
cd build
```
3. Run CMake to generate project files:
```
cmake -A x64 <open_model_zoo>/demos
```
4. Run `cmake --build` tool to  build the demos:
  - For release configuration
  ```
  cmake --build . --config Release
  ```
  - For debug configuration:
  ```
  cmake --build . --config Debug
  ```

### <a name="build_python_extensions"></a>Build the Native Python\* Extension Modules

Some of the Python demo applications require native Python extension modules to be built before they can be run.
This requires you to have Python development files (headers and import libraries) installed.
To build these modules, follow the instructions for building the demo applications above,
but add `-DENABLE_PYTHON=ON` to either the `cmake` or the `build_demos*` command, depending on which you use.
For example:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON <open_model_zoo>/demos
```

### <a name="build_specific_demos"></a>Build Specific Demos

To build specific demos, follow the instructions for building the demo applications above,
but add `--target <demo1> <demo2> ...` to the `cmake --build` command or `--target="<demo1> <demo2> ..."` to the `build_demos*` command.
Note, `cmake --build` tool supports multiple targets starting with version 3.15, with lower versions you can specify only one target.

For Linux*:
```sh
cmake -DCMAKE_BUILD_TYPE=Release <open_model_zoo>/demos
cmake --build . --target classification_demo segmentation_demo
```
or
```sh
build_demos.sh --target="classification_demo segmentation_demo"
```
For Microsoft Windows* OS:

```
cmake -A x64 <open_model_zoo>/demos
cmake --build . --config Release --target classification_demo segmentation_demo
```
or

```bat
build_demos_msvc.bat --target="classification_demo segmentation_demo"
```

## Get Ready for Running the Demo Applications

### Get Ready for Running the Demo Applications on Linux*

Before running compiled binary files, make sure your application can find the Inference Engine and OpenCV libraries.
If you use a [proprietary](https://software.intel.com/en-us/openvino-toolkit) distribution to build demos,
run the `setupvars` script to set all necessary environment variables:

```sh
source <INSTALL_DIR>/bin/setupvars.sh
```

If you use your own Inference Engine and OpenCV binaries to build the demos please make sure you have added them
to the `LD_LIBRARY_PATH` environment variable.

**(Optional)**: The OpenVINO environment variables are removed when you close the
shell. As an option, you can permanently set the environment variables as follows:

1. Open the `.bashrc` file in `<user_home_directory>`:

```sh
vi <user_home_directory>/.bashrc
```

2. Add this line to the end of the file:

```sh
source <INSTALL_DIR>/bin/setupvars.sh
```

3. Save and close the file: press the **Esc** key, type `:wq` and press the **Enter** key.
4. To test your change, open a new terminal. You will see `[setupvars.sh] OpenVINO environment initialized`.

To run Python demo applications that require native Python extension modules, you must additionally
set up the `PYTHONPATH` environment variable as follows, where `<bin_dir>` is the directory with
the built demo applications:

```sh
export PYTHONPATH="$PYTHONPATH:<bin_dir>/lib"
```

You are ready to run the demo applications. To learn about how to run a particular
demo, read the demo documentation by clicking the demo name in the demo
list above.

### Get Ready for Running the Demo Applications on Windows*

Before running compiled binary files, make sure your application can find the Inference Engine and OpenCV libraries.
Optionally download OpenCV community FFmpeg plugin. There is a downloader script in the OpenVINO package: `<INSTALL_DIR>\opencv\ffmpeg-download.ps1`.
If you use a [proprietary](https://software.intel.com/en-us/openvino-toolkit) distribution to build demos,
run the `setupvars` script to set all necessary environment variables:

```bat
<INSTALL_DIR>\bin\setupvars.bat
```

If you use your own Inference Engine and OpenCV binaries to build the demos please make sure you have added
to the `PATH` environment variable.

To run Python demo applications that require native Python extension modules, you must additionally
set up the `PYTHONPATH` environment variable as follows, where `<bin_dir>` is the directory with
the built demo applications:

```bat
set PYTHONPATH=%PYTHONPATH%;<bin_dir>
```

To debug or run the demos on Windows in Microsoft Visual Studio, make sure you
have properly configured **Debugging** environment settings for the **Debug**
and **Release** configurations. Set correct paths to the OpenCV libraries, and
debug and release versions of the Inference Engine libraries.
For example, for the **Debug** configuration, go to the project's
**Configuration Properties** to the **Debugging** category and set the `PATH`
variable in the **Environment** field to the following:

```
PATH=<INSTALL_DIR>\deployment_tools\inference_engine\bin\intel64\Debug;<INSTALL_DIR>\opencv\bin;%PATH%
```

where `<INSTALL_DIR>` is the directory in which the OpenVINO toolkit is installed.

You are ready to run the demo applications. To learn about how to run a particular
demo, read the demo documentation by clicking the demo name in the demos
list above.

## See Also

* [Intel OpenVINO Documentation](https://docs.openvinotoolkit.org/latest/documentation.html)
* [Overview of OpenVINO&trade; Toolkit Intel's Pre-Trained Models](../models/intel/index.md)
* [Overview of OpenVINO&trade; Toolkit Public Pre-Trained Models](../models/public/index.md)

---
\* Other names and brands may be claimed as the property of others.
