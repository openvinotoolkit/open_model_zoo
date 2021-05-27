# Face Recognition Python\* Demo

This example demonstrates an approach to create interactive applications
for video processing. It shows the basic architecture for building model
pipelines supporting model placement on different devices and simultaneous
parallel or sequential execution using OpenVINO library in Python.
In particular, this demo uses 3 models to build a pipeline able to detect
faces on videos, their keypoints (aka "landmarks"),
and recognize persons using the provided faces database (the gallery).
The following pretrained models can be used:

* `face-detection-retail-0004` and `face-detection-adas-0001`, to detect faces and predict their bounding boxes;
* `landmarks-regression-retail-0009`, to predict face keypoints;
* `face-reidentification-retail-0095`, to recognize persons.

## How it works

The application is invoked from command line. It reads the specified input
video stream frame-by-frame, be it a camera device or a video file,
and performs independent analysis of each frame. In order to make predictions
the application deploys 3 models on the specified devices using OpenVINO
library and runs them in asynchronous manner. An input frame is processed by
the face detection model to predict face bounding boxes. Then, face keypoints
are predicted by the corresponding model. The final step in frame processing
is done by the face recognition model, which uses keypoints found
to align the faces and the face gallery to match faces found on a video
frame with the ones in the gallery. Then, the processing results are
visualized and displayed on the screen or written to the output file.

## Preparing to Run

### Installation and dependencies

The demo depends on:

* OpenVINO library (2021.4 or newer)
* Python (any, which is supported by OpenVINO)
* OpenCV (>=4.2.5)

To install all the required Python modules you can use:

``` sh
pip install -r requirements.txt
```

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in <omz_dir>/demos/face_recognition_demo/python/models.lst file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

### Supported Models

* face-detection-adas-0001
* face-detection-retail-0004
* face-reidentification-retail-0095
* landmarks-regression-retail-0009
* Sphereface

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

### Creating a gallery for face recognition

To recognize faces the application uses a face database, or a gallery.
The gallery is a folder with images of persons. Each image in the gallery can
be of arbitrary size and should contain one or more frontally-oriented faces
with decent quality. There are allowed multiple images of the same person, but
the naming format in that case should be specific - `{id}-{num_of_instance}.jpg`.
For example, there could be images `Paul-0.jpg`, `Paul-1.jpg` etc.
and they all will be treated as images of the same person. In case when there
is one image per person, you can use format `{id}.jpg` (e.g. `Paul.jpg`).
The application can use face detector during the gallery building, that is
controlled by `--run_detector` flag. This allows gallery images to contain more
than one face image and not to be tightly cropped. In that mode the user will
be asked if he wants to add a specific image to the images gallery (and it
leads to automatic dumping images to the same folder on disk). If it is, then
the user should specify the name for the image in the open window and press
`Enter`. If it's not, then press `Escape`. The user may add multiple images of
the same person by setting the same name in the open window. However, the
resulting gallery needs to be checked more thoroughly, since a face detector can
fail and produce poor crops.

Image file name is used as a person name during the visualization.
Use the following name convention: `person_N_name.png` or `person_N_name.jpg`.

## Running

Running the application with the `-h` option or without
any arguments yields the following message:

```
python ./face_recognition_demo.py -h

usage: face_recognition_demo.py [-h] -i INPUT [--loop] [-o OUTPUT]
                                [-limit OUTPUT_LIMIT] [--no_show]
                                [--output_resolution OUTPUT_RESOLUTION]
                                [-cw CROP_WIDTH] [-ch CROP_HEIGHT]
                                [--match_algo {HUNGARIAN,MIN_DIST}]
                                [-u UTILIZATION_MONITORS]
                                -fg PATH [--run_detector] [--allow_grow]
                                -m_fd PATH -m_lm PATH -m_reid PATH
                                [-fd_iw FD_INPUT_WIDTH] [-fd_ih FD_INPUT_HEIGHT]
                                [-d_fd {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}]
                                [-d_lm {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}]
                                [-d_reid {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}]
                                [-l PATH] [-c PATH] [-v] [-pc] [-t_fd [0..1]]
                                [-t_id [0..1]] [-exp_r_fd NUMBER]

Optional arguments:
  -h, --help            Show this help message and exit.

General:
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera id.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output.
                        If 0 is set, all frames are stored.
  --output_resolution OUTPUT_RESOLUTION
                        Optional. Specify the maximum output window resolution
                        in (width x height) format. Example: 1280x720.
                        Input frame size used by default.
  --no_show             Optional. Don't show output.
  -cw CROP_WIDTH, --crop_width CROP_WIDTH
                        Optional. Crop the input stream to this width.
                        Both -cw and -ch parameters should be specified
                        to use crop.
  -ch CROP_HEIGHT, --crop_height CROP_HEIGHT
                        Optional. Crop the input stream to this height.
                        Both -cw and -ch parameters should be specified
                        to use crop.
  --match_algo {HUNGARIAN,MIN_DIST}
                        Optional. Algorithm for face matching.
                        Default: HUNGARIAN.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.

Faces database:
  -fg                   Required. Path to the face images directory.
  --run_detector        Optional. Use Face Detection model to find faces
                        on the face images, otherwise use full images.
  --allow_grow          Optional. Flag to allow growing the face database,
                        in addition allow dumping new faces on disk. In that
                        case the user will be asked if he wants to add a
                        specific image to the images gallery (and it leads to
                        automatic dumping images to the same folder on disk).
                        If it is, then the user should specify the name for
                        the image in the open window and press `Enter`.
                        If it's not, then press `Escape`. The user may add
                        new images for the same person by setting the same
                        name in the open window.

Models:
  -m_fd PATH            Required. Path to an .xml file with Face Detection model.
  -m_lm PATH            Required. Path to an .xml file with Facial Landmarks Detection
                        model.
  -m_reid PATH          Required. Path to an .xml file with Face Reidentification
                        model.
  -fd_iw FD_INPUT_WIDTH, --fd_input_width FD_INPUT_WIDTH
                        Optional. Specify the input width of detection model.
                        Both -fd_iw and -fd_ih parameters should be specified
                        for reshape.
  -fd_ih FD_INPUT_HEIGHT, --fd_input_height FD_INPUT_HEIGHT
                        Optional. Specify the input height of detection model.
                        Both -fd_iw and -fd_ih parameters should be specified
                        for reshape.

Inference options:
  -d_fd {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}
                        Optional. Target device for Face Detection model.
                        Default value is CPU.
  -d_lm {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}
                        Optional. Target device for Facial Landmarks Detection
                        model. Default value is CPU.
  -d_reid {CPU,GPU,FPGA,MYRIAD,HETERO,HDDL}
                        Optional. Target device for Face Reidentification
                        model. Default value is CPU.
  -l PATH, --cpu_lib PATH
                        Optional. For MKLDNN (CPU)-targeted custom layers,
                        if any. Path to a shared library with custom
                        layers implementations.
  -c PATH, --gpu_lib PATH
                        Optional. For clDNN (GPU)-targeted custom layers,
                        if any. Path to the XML file with descriptions
                        of the kernels.
  -v, --verbose         Optional. Be more verbose.
  -pc, --perf_stats     Optional. Output detailed per-layer performance stats.
  -t_fd [0..1]          Optional. Probability threshold for face detections.
  -t_id [0..1]          Optional. Cosine distance threshold between two vectors
                        for face identification.
  -exp_r_fd NUMBER      Optional. Scaling ratio for bboxes passed to face
                        recognition.
```

Example of a valid command line to run the application:

Linux (`sh`, `bash`, ...) (assuming OpenVINO installed in `/opt/intel/openvino`):

``` sh
# Set up the environment
source /opt/intel/openvino/bin/setupvars.sh

python ./face_recognition_demo.py \
-m_fd <path_to_model>/face-detection-retail-0004.xml \
-m_lm <path_to_model>/landmarks-regression-retail-0009.xml \
-m_reid <path_to_model>/face-reidentification-retail-0095.xml \
--verbose \
-fg "/home/face_gallery"
```

Windows (`cmd`, `powershell`) (assuming OpenVINO installed in `C:/Intel/openvino`):

```bat
# Set up the environment
call C:/Intel/openvino/bin/setupvars.bat

python ./face_recognition_demo.py ^
-m_fd <path_to_model>/face-detection-retail-0004.xml ^
-m_lm <path_to_model>/landmarks-regression-retail-0009.xml ^
-m_reid <path_to_model>/face-reidentification-retail-0095.xml ^
--verbose ^
-fg "C:/face_gallery"
```

## Demo output

The demo uses OpenCV window to display the resulting video frame and detections.
If specified, it also writes output to a file. It outputs logs to the terminal.

## See also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
