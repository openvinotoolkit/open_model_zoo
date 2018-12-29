# Interactive Face Recognition Demo

This example demonstrates an approach to create interactive applications
for video processing. It shows the basic architecture for building model
pipelines supporting model placement on different devices and simultaneous
parallel or sequential execution using OpenVINO library in Python.
In particular, this demo uses 4 models to build a pipeline able to detect
faces on videos, detect face direction and its keypoints (aka "landmarks"),
and recognize persons using provided face database (gallery). The corresponding
pretrained models can be found in this repository:

* `intel_models/face-detection-retail-0004` and
  `intel_models/face-detection-adas-0001`,
    which are used to detect faces and predict their bounding boxes;
* `intel_models/head-pose-estimation-adas-0001`,
    which is used to predict head direction;
* `intel_models/landmarks-regression-retail-0009`,
    which is used to predict face keypoints;
* `intel_models/face-reidentification-retail-0095`,
    which is used to recognize persons.

### How it works

The application is invoked from command line. It reads the specified input
video stream frame-by-frame, be it a camera device or a video file,
and performs independent analysis of each frame. In order to make predictions
the application deploys 4 models on the specified devices using OpenVINO
library and runs them in asynchronous manner. The first model to be used
is the face detection model, which is followed by two independent models
for head pose prediction and face keypoints prediction. The last step in
frame processing is done by face recognition model, which uses keypoints
to align the faces and the face gallery to match faces found on a video
frame with the ones in the gallery. Then, the processing results are
visualized and displayed on the screen or written to the output file.

### Creating a gallery for face recognition

In order to recognize faces the application needs a face database, or a gallery.
The gallery is a folder with images of persons. Each image in the gallery can
be of arbitrary size and should contain a tight crop of face. To obtain better
results use images of square shapes. Image file name is used as a person name
during the visualization.
Use the following name convention: `person_N_name.png` or `person_N_name.jpg`.

### Installation and dependencies

The demo depends on:
- OpenVINO library (R4)
- Python (any of 2.7+ or 3.4+, which is supported by OpenVINO)
- NumPy (>=1.11.0)
- SciPy (>=1.1.0)
- OpenCV (>=3.4.0)

To install all the required Python modules you can use:

``` sh
pip install -r requirements.txt
```

### Running the demo:

Running the application with the `-h` option or without
any arguments yields the following message:

``` sh
./face_recognition_demo.py -h

usage: face_recognition_demo.py [-h] [-i PATH] [-o PATH] [-no_show]
                                [-cw CROP_WIDTH] [-ch CROP_HEIGHT] -fg PATH
                                -m_fd PATH -m_lm PATH -m_reid PATH -m_hp PATH
                                [-d_fd {CPU,GPU,FPGA,MYRIAD,HETERO}]
                                [-d_lm {CPU,GPU,FPGA,MYRIAD,HETERO}]
                                [-d_reid {CPU,GPU,FPGA,MYRIAD,HETERO}]
                                [-d_hp {CPU,GPU,FPGA,MYRIAD,HETERO}] [-l PATH]
                                [-c PATH] [-v] [-pc] [-t_fd [0..1]]
                                [-t_id [0..1]] [-exp_r_fd NUMBER]

optional arguments:
  -h, --help            show this help message and exit

General:
  -i PATH, --input PATH
                        (optional) Path to the input video ('cam' for the
                        camera, default)
  -o PATH, --output PATH
                        (optional) Path to save the output video to
  -no_show              (optional) Do not display output
  -cw CROP_WIDTH, --crop_width CROP_WIDTH
                        (optional) Crop the input stream to this width
  -ch CROP_HEIGHT, --crop_height CROP_HEIGHT
                        (optional) Crop the input stream to this height

Faces database:
  -fg PATH              Path to the face images directory

Models:
  -m_fd PATH            Path to the Face Detection Adas or Retail model XML
                        file
  -m_lm PATH            Path to the Facial Landmarks Regression Retail model
                        XML file
  -m_reid PATH          Path to the Face Reidentification Retail model XML
                        file
  -m_hp PATH            Path to the Head Pose Estimation Retail model XML file

Inference options:
  -d_fd {CPU,GPU,FPGA,MYRIAD,HETERO}
                        (optional) Target device for the Face Detection Retail
                        model (default: CPU)
  -d_lm {CPU,GPU,FPGA,MYRIAD,HETERO}
                        (optional) Target device for the Facial Landmarks
                        Regression Retail model (default: CPU)
  -d_reid {CPU,GPU,FPGA,MYRIAD,HETERO}
                        (optional) Target device for the Face Reidentification
                        Retail model (default: CPU)
  -d_hp {CPU,GPU,FPGA,MYRIAD,HETERO}
                        (optional) Target device for the Head Pose Estimation
                        Retail model (default: CPU)
  -l PATH, --cpu_lib PATH
                        (optional) For MKLDNN (CPU)-targeted custom layers, if
                        any. Path to a shared library with custom layers
                        implementations
  -c PATH, --gpu_lib PATH
                        (optional) For clDNN (GPU)-targeted custom layers, if
                        any. Path to the XML file with descriptions of the
                        kernels
  -v, --verbose         (optional) Be more verbose
  -pc, --perf_stats     (optional) Output detailed per-layer performance stats
  -t_fd [0..1]          (optional) Probability threshold for face
                        detections(default: 0.6)
  -t_id [0..1]          (optional) Cosine distance threshold between two
                        vectors for face identification(default: 0.3)
  -exp_r_fd NUMBER      (optional) Scaling ratio for bbox passed to face
                        recognition(default: 1.15)
```

Example of a valid command line to run the application:

Linux (`sh`, `bash`, ...) (assuming OpenVINO installed in `/opt/intel/computer_vision_sdk`):

``` sh
# Set up the environment
source /opt/intel/computer_vision_sdk/bin/setupvars.sh

./face_recognition_demo.py \
-m_fd /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml \
-m_hp /opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml \
-m_lm /opt/intel/computer_vision_sdk/deployment_tools/intel_models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
-m_reid /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-reidentification-retail-0071/FP32/face-reidentification-retail-0071.xml \
-l /opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so \
--verbose \
-fg "/home/face_gallery"
```

Windows (`cmd`, `powershell`) (assuming OpenVINO installed in `C:/Intel/computer_vision_sdk`):

``` powershell
# Set up the environment
C:/Intel/computer_vision_sdk/bin/setupvars.bat

python ./face_recognition_demo.py ^
-m_fd C:/Intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml ^
-m_hp C:/Intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml ^
-m_lm C:/Intel/computer_vision_sdk/deployment_tools/intel_models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml ^
-m_reid C:/Intel/computer_vision_sdk/deployment_tools/intel_models/face-reidentification-retail-0071/FP32/face-reidentification-retail-0071.xml ^
-l C:/Intel/computer_vision_sdk/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll ^
--verbose ^
-fg "C:/face_gallery"
```

Notice that the custom networks should be converted to the
Inference Engine format (*.xml + *bin) first. To do this use the
[Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) tool.

### Demo output

The demo uses OpenCV window to display the resulting video frame and detections.
If specified, it also writes output to a file. It outputs logs to the terminal.

## See also
* [Using Inference Engine Demos](../Readme.md)