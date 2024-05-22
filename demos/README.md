# Open Model Zoo Demos

<!--
@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   omz_demos_human_pose_estimation_3d_demo_python
   omz_demos_3d_segmentation_demo_python
   omz_demos_action_recognition_demo_python
   omz_demos_background_subtraction_demo_cpp_gapi
   omz_demos_background_subtraction_demo_python
   omz_demos_bert_named_entity_recognition_demo_python
   omz_demos_bert_question_answering_embedding_demo_python
   omz_demos_bert_question_answering_demo_python
   omz_demos_classification_benchmark_demo_cpp
   omz_demos_classification_benchmark_demo_cpp_gapi
   omz_demos_classification_demo_python
   omz_demos_colorization_demo_python
   omz_demos_crossroad_camera_demo_cpp
   omz_demos_face_recognition_demo_python
   omz_demos_formula_recognition_demo_python
   omz_demos_gaze_estimation_demo_cpp_gapi
   omz_demos_interactive_face_detection_demo_cpp_gapi
   omz_demos_gaze_estimation_demo_cpp
   omz_demos_gesture_recognition_demo_cpp_gapi
   omz_demos_gesture_recognition_demo_python
   omz_demos_gpt2_text_prediction_demo_python
   omz_demos_handwritten_text_recognition_demo_python
   omz_demos_human_pose_estimation_demo_cpp
   omz_demos_human_pose_estimation_demo_python
   omz_demos_image_inpainting_demo_python
   omz_demos_image_processing_demo_cpp
   omz_demos_image_retrieval_demo_python
   omz_demos_segmentation_demo_cpp
   omz_demos_segmentation_demo_python
   omz_demos_image_translation_demo_python
   omz_demos_instance_segmentation_demo_python
   omz_demos_interactive_face_detection_demo_cpp
   omz_demos_machine_translation_demo_python
   omz_demos_monodepth_demo_python
   omz_demos_mri_reconstruction_demo_cpp
   omz_demos_mri_reconstruction_demo_python
   omz_demos_multi_camera_multi_target_tracking_demo_python
   omz_demos_multi_channel_face_detection_demo_cpp
   omz_demos_multi_channel_human_pose_estimation_demo_cpp
   omz_demos_multi_channel_object_detection_demo_yolov3_cpp
   omz_demos_noise_suppression_demo_cpp
   omz_demos_noise_suppression_demo_python
   omz_demos_object_detection_demo_cpp
   omz_demos_object_detection_demo_python
   omz_demos_pedestrian_tracker_demo_cpp
   omz_demos_place_recognition_demo_python
   omz_demos_security_barrier_camera_demo_cpp
   omz_demos_single_human_pose_estimation_demo_python
   omz_demos_smartlab_demo_python
   omz_demos_smart_classroom_demo_cpp
   omz_demos_smart_classroom_demo_cpp_gapi
   omz_demos_social_distance_demo_cpp
   omz_demos_sound_classification_demo_python
   omz_demos_speech_recognition_deepspeech_demo_python
   omz_demos_speech_recognition_quartznet_demo_python
   omz_demos_speech_recognition_wav2vec_demo_python
   omz_demos_mask_rcnn_demo_cpp
   omz_demos_text_detection_demo_cpp
   omz_demos_text_spotting_demo_python
   omz_demos_text_to_speech_demo_python
   omz_demos_time_series_forecasting_demo_python
   omz_demos_whiteboard_inpainting_demo_python

@endsphinxdirective
-->

Open Model Zoo demos are console applications that provide templates to help implement specific deep learning inference scenarios. These applications show how to preprocess and postrpocess data for model inference and organize processing pipelines. Some pipelines collect analysis data from several models being inferred simultaneously. For example, [detecting a person in a video stream along with detecting the person's physical attributes, such as age, gender, and emotional state](./interactive_face_detection_demo/cpp/README.md).

Source code of the demos can be obtained from the Open Model Zoo [GitHub repository](https://github.com/openvinotoolkit/open_model_zoo/).

```sh
git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
```

C++, C++ G-API and Python\* versions are located in the `cpp`, `cpp_gapi` and `python` subdirectories respectively.

The Open Model Zoo includes the following demos:

- [3D Human Pose Estimation Python\* Demo](./human_pose_estimation_3d_demo/python/README.md) - 3D human pose estimation demo.
- [3D Segmentation Python\* Demo](./3d_segmentation_demo/python/README.md) - Segmentation demo segments 3D images using 3D convolutional networks.
- [Action Recognition Python\* Demo](./action_recognition_demo/python/README.md) - Demo application for Action Recognition algorithm, which classifies actions that are being performed on input video.
- [Background Subtraction Python\* Demo](./background_subtraction_demo/python/README.md) - Background subtraction using instance segmentation based models.
- [Background Subtraction C++ G-API\* Demo](./background_subtraction_demo/cpp_gapi/README.md) - Background subtraction G-API version.
- [BERT Named Entity Recognition Python\* Demo](./bert_named_entity_recognition_demo/python/README.md) - NER Demo application that uses a CONLL2003-tuned BERT model for inference.
- [BERT Question Answering Python\* Demo](./bert_question_answering_demo/python/README.md)
- [BERT Question Answering Embedding Python\* Demo](./bert_question_answering_embedding_demo/python/README.md) - The demo demonstrates how to run BERT based models for question answering task.
- [Classification Python\* Demo](./classification_demo/python/README.md) - Shows an example of using neural networks for image classification.
- [Classification Benchmark C++ Demo](./classification_benchmark_demo/cpp/README.md) - Visualizes OpenVINO performance on inference of neural networks for image classification.
- [Classification Benchmark C++ G-API Demo](./classification_benchmark_demo/cpp_gapi/README.md) - Classification Benchmark C++ G-API version.
- [Colorization Python\* Demo](./colorization_demo/python/README.md) - Colorization demo colorizes input frames.
- [Crossroad Camera C++ Demo](./crossroad_camera_demo/cpp/README.md) - Person Detection followed by the Person Attributes Recognition and Person Reidentification Retail, supports images/video and camera inputs.
- [Face Recognition Python\* Demo](./face_recognition_demo/python/README.md) - The interactive face recognition demo.
- [Formula Recognition Python\* Demo](./formula_recognition_demo/python/README.md) - The demo demonstrates how to run Im2latex formula recognition models and recognize latex formulas.
- [Gaze Estimation C++ Demo](./gaze_estimation_demo/cpp/README.md) - Face detection followed by gaze estimation, head pose estimation and facial landmarks regression.
- [Gaze Estimation C++ G-API\* Demo](./gaze_estimation_demo/cpp_gapi/README.md) - Face detection followed by gaze estimation, head pose estimation and facial landmarks regression. G-API version.
- [Gesture Recognition Python\* Demo](./gesture_recognition_demo/python/README.md) - Demo application for Gesture Recognition algorithm (e.g. American Sign Language gestures), which classifies gesture actions that are being performed on input video.
- [Gesture Recognition C++ G-API\* Demo](./gesture_recognition_demo/cpp_gapi/README.md) - Demo application for Gesture Recognition algorithm (e.g. American Sign Language gestures), which classifies gesture actions that are being performed on input video. G-API version.
- [GPT-2 Text Prediction Python\* Demo](./gpt2_text_prediction_demo/python/README.md) - GPT-2 text prediction demo.
- [Handwritten Text Recognition Python\* Demo](./handwritten_text_recognition_demo/python/README.md) - The demo demonstrates how to run Handwritten Text Recognition models for Japanese, Simplified Chinese and English.
- [Human Pose Estimation C++ Demo](./human_pose_estimation_demo/cpp/README.md) - Human pose estimation demo.
- [Human Pose Estimation Python\* Demo](./human_pose_estimation_demo/python/README.md) - Human pose estimation demo.
- [Image Inpainting Python\* Demo](./image_inpainting_demo/python/README.md) - Demo application for GMCNN inpainting network.
- [Image Processing C++ Demo](./image_processing_demo/cpp/README.md) - Demo application for enhancing the resolution of the input image.
- [Image Retrieval Python\* Demo](./image_retrieval_demo/python/README.md) - The demo demonstrates how to run Image Retrieval models using OpenVINO&trade;.
- [Image Segmentation C++ Demo](./segmentation_demo/cpp/README.md) - Inference of semantic segmentation networks (supports video and camera inputs).
- [Image Segmentation Python\* Demo](./segmentation_demo/python/README.md) - Inference of semantic segmentation networks (supports video and camera inputs).
- [Image Translation Python\* Demo](./image_translation_demo/python/README.md) - Demo application to synthesize a photo-realistic image based on exemplar image.
- [Instance Segmentation Python\* Demo](./instance_segmentation_demo/python/README.md) - Inference of instance segmentation networks trained in `Detectron` or `maskrcnn-benchmark`.
- [Interactive Face Detection C++ Demo](./interactive_face_detection_demo/cpp/README.md) - Face Detection coupled with Age/Gender, Head-Pose, Emotion, and Facial Landmarks detectors. Supports video and camera inputs.
- [Interactive Face Detection G-API\* Demo](./interactive_face_detection_demo/cpp_gapi/README.md) - G-API based Face Detection coupled with Age/Gender, Head-Pose, Emotion, and Facial Landmarks detectors. Supports video and camera inputs.
- [Machine Translation Python\* Demo](./machine_translation_demo/python/README.md) - The demo demonstrates how to run non-autoregressive machine translation models.
- [Mask R-CNN C++ Demo for TensorFlow\* Object Detection API](./mask_rcnn_demo/cpp/README.md) - Inference of instance segmentation networks created with TensorFlow\* Object Detection API.
- [Monodepth Python\* Demo](./monodepth_demo/python/README.md) - The demo demonstrates how to run monocular depth estimation models.
- [MRI Reconstruction C++ Demo](./mri_reconstruction_demo/cpp/README.md) - Compressed Sensing demo for medical images
- [MRI Reconstruction Python\* Demo](./mri_reconstruction_demo/python/README.md) - Compressed Sensing demo for medical images
- [Multi-Camera Multi-Target Tracking Python\* Demo](./multi_camera_multi_target_tracking_demo/python/README.md) Demo application for multiple targets (persons or vehicles) tracking on multiple cameras.
- [Multi-Channel Face Detection C++ Demo](./multi_channel_face_detection_demo/cpp/README.md) - The demo demonstrates an inference pipeline for multi-channel face detection scenario.
- [Multi-Channel Human Pose Estimation C++ Demo](./multi_channel_human_pose_estimation_demo/cpp/README.md) - The demo demonstrates an inference pipeline for multi-channel human pose estimation scenario.
- [Multi-Channel Object Detection Yolov3 C++ Demo](./multi_channel_object_detection_demo_yolov3/cpp/README.md) - The demo demonstrates an inference pipeline for multi-channel common object detection scenario.
- [Noise Suppression Python\* Demo](./noise_suppression_demo/python/README.md) - The demo shows how to use the OpenVINO™ toolkit to reduce noise in speech audio.
- [Noise Suppression C++\* Demo](./noise_suppression_demo/cpp/README.md) - The demo shows how to use the OpenVINO™ toolkit to reduce noise in speech audio.
- [Object Detection Python\* Demo](./object_detection_demo/python/README.md) - Demo application for several object detection model types (like SSD, Yolo, etc).
- [Object Detection C++ Demo](./object_detection_demo/cpp/README.md) - Demo application for Object Detection networks (different models architectures are supported), async API showcase, simple OpenCV interoperability (supports video and camera inputs).
- [Pedestrian Tracker C++ Demo](./pedestrian_tracker_demo/cpp/README.md) - Demo application for pedestrian tracking scenario.
- [Place Recognition Python\* Demo](./place_recognition_demo/python/README.md) - This demo demonstrates how to run Place Recognition models using OpenVINO™.
- [Security Barrier Camera C++ Demo](./security_barrier_camera_demo/cpp/README.md) - Vehicle Detection followed by the Vehicle Attributes and License-Plate Recognition, supports images/video and camera inputs.
- [Speech Recognition DeepSpeech Python\* Demo](./speech_recognition_deepspeech_demo/python/README.md) - Speech recognition demo: accepts an audio file with an English phrase on input and converts it into text. This demo does streaming audio data processing and can optionally provide current transcription of the processed part.
- [Speech Recognition QuartzNet Python\* Demo](./speech_recognition_quartznet_demo/python/README.md) - Speech recognition demo for QuartzNet: takes a whole audio file with an English phrase on input and converts it into text.
- [Speech Recognition Wav2Vec Python\* Demo](./speech_recognition_wav2vec_demo/python/README.md) - Speech recognition demo for Wav2Vec: takes a whole audio file with an English phrase on input and converts it into text.
- [Single Human Pose Estimation Python\* Demo](./single_human_pose_estimation_demo/python/README.md) - 2D human pose estimation demo.
- [Smart Classroom C++ Demo](./smart_classroom_demo/cpp/README.md) - Face recognition and action detection demo for classroom environment.
- [Smart Classroom C++ G-API Demo](./smart_classroom_demo/cpp_gapi/README.md) - Face recognition and action detection demo for classroom environment. G-PI version.
- [Smartlab Python\* Demo](./smartlab_demo/python/README.md) - action recognition and object detection for smartlab.
- [Social Distance C++ Demo](./social_distance_demo/cpp/README.md) - This demo showcases a retail social distance application that detects people and measures the distance between them.
- [Sound Classification Python\* Demo](./sound_classification_demo/python/README.md) - Demo application for sound classification algorithm.
- [Text Detection C++ Demo](./text_detection_demo/cpp/README.md) - Text Detection demo. It detects and recognizes multi-oriented scene text on an input image and puts a bounding box around detected area.
- [Text Spotting Python\* Demo](./text_spotting_demo/python/README.md) - The demo demonstrates how to run Text Spotting models.
- [Text-to-speech Python\* Demo](./text_to_speech_demo/python/README.md) - Shows an example of using Forward Tacotron and WaveRNN neural networks for text to speech task.
- [Time Series Forecasting Python\* Demo](./time_series_forecasting_demo/python/README.md) - The demo shows how to use the OpenVINO™ toolkit to time series forecasting.
- [Whiteboard Inpainting Python\* Demo](./whiteboard_inpainting_demo/python/README.md) - The demo shows how to use the OpenVINO™ toolkit to detect and hide a person on a video so that all text on a whiteboard is visible.

## Media Files Available for Demos

To run the demo applications, you can use videos from https://storage.openvinotoolkit.org/data/test_data/videos.

## Demos that Support Pre-Trained Models

You can download the [Intel pre-trained models](../models/intel/index.md) or [public pre-trained models](../models/public/index.md) using the OpenVINO [Model Downloader](../tools/model_tools/README.md).

## Build the Demo Applications

To build the demos, you need to source OpenVINO™ environment and [get OpenCV](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO). You can install the OpenVINO™ toolkit using the installation package for [Intel® Distribution of OpenVINO™ toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit-download.html) or build the open-source version available in the [OpenVINO GitHub repository](https://github.com/openvinotoolkit/openvino) using the [build instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).
For the Intel® Distribution of OpenVINO™ toolkit installed to the `<INSTALL_DIR>` directory on your machine, run the following commands to download prebuilt OpenCV and set environment variables before building the demos:

```sh
source <INSTALL_DIR>/setupvars.sh
```

> **NOTE:** If you plan to use Python\* demos only, you can install the OpenVINO Python\* package.
> ```sh
> pip install openvino
> ```

For the open-source version of OpenVINO, set the following variables:
* `OpenVINO_DIR` pointing to a folder containing `OpenVINOConfig.cmake`
* `OpenCV_DIR` pointing to OpenCV. The same OpenCV version should be used both for OpenVINO and demos build.

Alternatively, these values can be provided via command line while running `cmake`. See [CMake search procedure](https://cmake.org/cmake/help/latest/command/find_package.html#search-procedure).
Also add paths to the built OpenVINO™ Runtime libraries to the `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows) variable before building the demos.

### Build the Demo Applications on Linux*

The officially supported Linux* build environment is the following:

- Ubuntu* 18.04 LTS 64-bit or Ubuntu* 20.04 LTS 64-bit
- GCC* 7.5.0 (for Ubuntu* 18.04) or GCC* 9.3.0 (for Ubuntu* 20.04)
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

### Build the Demos Applications on Microsoft Windows* OS

The recommended Windows* build environment is the following:

- Microsoft Windows* 10
- Microsoft Visual Studio* 2019
- CMake* version 3.14 or higher

To build the demo applications for Windows, go to the directory with the `build_demos_msvc.bat`
batch file and run it:

```bat
build_demos_msvc.bat
```

By default, the script automatically detects the highest Microsoft Visual Studio version installed on the machine and uses it to create and build
a solution for a demo code. Optionally, you can also specify the preferred Microsoft Visual Studio version to be used by the script. Supported
version is: `VS2019`. For example, to build the demos using the Microsoft Visual Studio 2019, use the following command:

```bat
build_demos_msvc.bat VS2019
```

By default, the demo applications binaries are build into the `C:\Users\<username>\Documents\Intel\OpenVINO\omz_demos_build\intel64\Release` directory.
The default build folder can be changed with `-b` option. For example, following command will build Open Model Zoo demos into `c:\temp\omz-demos-build` folder:

```bat
build_demos_msvc.bat -b c:\temp\omz-demos-build
```


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

### Dependencies for Python* Demos

The dependencies for Python demos must be installed before running. It can be achieved with the following command:

```sh
python -mpip install --user -r <omz_dir>/demos/requirements.txt
```

### Python\* model API package

Python* ModelAPI is factored out as a separate package. Refer to the
[Python Model API documentation](./common/python/model_zoo/model_api/README.md#installing-python-model-api-package)
to learn about its installation. At the same time demos can find this package on their own. It's not required to install ModelAPI for demos.

###Build the Native Python\* Extension Modules

Some of the Python demo applications require native Python extension modules to be built before they can be run.
This requires you to have Python development files (headers and import libraries) installed.
To build these modules, follow the instructions for building the demo applications above,
but add `-DENABLE_PYTHON=ON` to either the `cmake` or the `build_demos*` command, depending on which you use.
For example:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON <open_model_zoo>/demos
```

Once the modules are built, add the demo build folder to the `PYTHONPATH` environment variable.

### Build Specific Demos

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

Before running compiled binary files, make sure your application can find the OpenVINO™ and OpenCV libraries.
If you use a [proprietary](https://software.intel.com/en-us/openvino-toolkit) distribution to build demos,
run the `setupvars` script to set all necessary environment variables:

```sh
source <INSTALL_DIR>/setupvars.sh
```

If you use your own OpenVINO™ and OpenCV binaries to build the demos please make sure you have added them
to the `LD_LIBRARY_PATH` environment variable.

**(Optional)**: The OpenVINO environment variables are removed when you close the
shell. As an option, you can permanently set the environment variables as follows:

1. Open the `.bashrc` file in `<user_home_directory>`:

```sh
vi <user_home_directory>/.bashrc
```

2. Add this line to the end of the file:

```sh
source <INSTALL_DIR>/setupvars.sh
```

3. Save and close the file: press the **Esc** key, type `:wq` and press the **Enter** key.
4. To test your change, open a new terminal. You will see `[setupvars.sh] OpenVINO environment initialized`.

To run Python demo applications that require native Python extension modules, you must additionally
set up the `PYTHONPATH` environment variable as follows, where `<bin_dir>` is the directory with
the built demo applications:

```sh
export PYTHONPATH="<bin_dir>:$PYTHONPATH"
```

You are ready to run the demo applications. To learn about how to run a particular
demo, read the demo documentation by clicking the demo name in the demo
list above.

### Get Ready for Running the Demo Applications on Windows*

Before running compiled binary files, make sure your application can find the OpenVINO™ and OpenCV libraries.
Optionally, download the OpenCV community FFmpeg plugin using the downloader script in the OpenVINO package: `<INSTALL_DIR>\extras\opencv\ffmpeg-download.ps1`.
If you use the [Intel® Distribution of OpenVINO™ toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) distribution to build demos,
run the `setupvars` script to set all necessary environment variables:

```bat
<INSTALL_DIR>\setupvars.bat
```

If you use your own OpenVINO™ and OpenCV binaries to build the demos please make sure you have added
to the `PATH` environment variable.

To run Python demo applications that require native Python extension modules, you must additionally
set up the `PYTHONPATH` environment variable as follows, where `<bin_dir>` is the directory with
the built demo applications:

```bat
set PYTHONPATH=<bin_dir>;%PYTHONPATH%
```

To debug or run the demos on Windows in Microsoft Visual Studio, make sure you
have properly configured **Debugging** environment settings for the **Debug**
and **Release** configurations. Set correct paths to the OpenCV libraries, and
debug and release versions of the OpenVINO™ libraries.
For example, for the **Debug** configuration, go to the project's
**Configuration Properties** to the **Debugging** category and set the `PATH`
variable in the **Environment** field to the following:

```
PATH=<INSTALL_DIR>\runtime\bin\intel64\Debug;<INSTALL_DIR>\extras\opencv\bin;%PATH%
```

where `<INSTALL_DIR>` is the directory in which the OpenVINO toolkit is installed.

You are ready to run the demo applications. To learn about how to run a particular
demo, read the demo documentation by clicking the demo name in the demos
list above.

## See Also

* [Intel OpenVINO Documentation](https://docs.openvino.ai/2023.0/documentation.html)
* [Overview of OpenVINO&trade; Toolkit Intel's Pre-Trained Models](../models/intel/index.md)
* [Overview of OpenVINO&trade; Toolkit Public Pre-Trained Models](../models/public/index.md)

---
\* Other names and brands may be claimed as the property of others.
