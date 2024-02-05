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
* `face-reidentification-retail-0095`, `facenet-20180408-102900` or `face-recognition-resnet100-arcface-onnx` to recognize persons.

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

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Embedding Preprocessing Computation](@ref openvino_docs_MO_DG_Additional_Optimization_Use_Cases).

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/face_recognition_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO IR format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models

* face-detection-adas-0001
* face-detection-retail-0004
* face-recognition-resnet100-arcface-onnx
* face-reidentification-retail-0095
* facenet-20180408-102900
* landmarks-regression-retail-0009

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
usage: face_recognition_demo.py [-h] -i INPUT [--loop] [-o OUTPUT] [-limit OUTPUT_LIMIT] [--output_resolution OUTPUT_RESOLUTION] [--no_show] [--crop_size CROP_SIZE CROP_SIZE] [--match_algo {HUNGARIAN,MIN_DIST}] [-u UTILIZATION_MONITORS]
                                [-fg FG] [--run_detector] [--allow_grow] -m_fd M_FD -m_lm M_LM -m_reid M_REID [--fd_input_size FD_INPUT_SIZE FD_INPUT_SIZE] [-d_fd {CPU,GPU,HETERO}] [-d_lm {CPU,GPU,HETERO}] [-d_reid {CPU,GPU,HETERO}] [-v]
                                [-t_fd [0..1]] [-t_id [0..1]] [-exp_r_fd NUMBER]

optional arguments:
  -h, --help            show this help message and exit

General:
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output. If 0 is set, all frames are stored.
  --output_resolution OUTPUT_RESOLUTION
                        Optional. Specify the maximum output window resolution in (width x height) format. Example: 1280x720. Input frame size used by default.
  --no_show             Optional. Don't show output.
  --crop_size CROP_SIZE CROP_SIZE
                        Optional. Crop the input stream to this resolution.
  --match_algo {HUNGARIAN,MIN_DIST}
                        Optional. Algorithm for face matching. Default: HUNGARIAN.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.

Faces database:
  -fg FG                Optional. Path to the face images directory.
  --run_detector        Optional. Use Face Detection model to find faces on the face images, otherwise use full images.
  --allow_grow          Optional. Allow to grow faces gallery and to dump on disk. Available only if --no_show option is off.

Models:
  -m_fd M_FD            Required. Path to an .xml file with Face Detection model.
  -m_lm M_LM            Required. Path to an .xml file with Facial Landmarks Detection model.
  -m_reid M_REID        Required. Path to an .xml file with Face Reidentification model.
  --fd_input_size FD_INPUT_SIZE FD_INPUT_SIZE
                        Optional. Specify the input size of detection model for reshaping. Example: 500 700.

Inference options:
  -d_fd {CPU,GPU,HETERO}
                        Optional. Target device for Face Detection model. Default value is CPU.
  -d_lm {CPU,GPU,HETERO}
                        Optional. Target device for Facial Landmarks Detection model. Default value is CPU.
  -d_reid {CPU,GPU,HETERO}
                        Optional. Target device for Face Reidentification model. Default value is CPU.
  -v, --verbose         Optional. Be more verbose.
  -t_fd [0..1]          Optional. Probability threshold for face detections.
  -t_id [0..1]          Optional. Cosine distance threshold between two vectors for face identification.
  -exp_r_fd NUMBER      Optional. Scaling ratio for bboxes passed to face recognition.
```

Example of a valid command line to run the application:

Linux (`sh`, `bash`, ...) (assuming OpenVINO installed in `/opt/intel/openvino`):

``` sh
# Set up the environment
source /opt/intel/openvino/bin/setupvars.sh

python ./face_recognition_demo.py \
  -i <path_to_video>/input_video.mp4 \
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
  -i <path_to_video>/input_video.mp4 ^
  -m_fd <path_to_model>/face-detection-retail-0004.xml ^
  -m_lm <path_to_model>/landmarks-regression-retail-0009.xml ^
  -m_reid <path_to_model>/face-reidentification-retail-0095.xml ^
  --verbose ^
  -fg "C:/face_gallery"
```

>**NOTE**: If you provide a single image as an input, the demo processes and renders it quickly, then exits. To continuously visualize inference results on the screen, apply the `loop` option, which enforces processing a single image in a loop.

You can save processed results to a Motion JPEG AVI file or separate JPEG or PNG files using the `-o` option:

* To save processed results in an AVI file, specify the name of the output file with `avi` extension, for example: `-o output.avi`.
* To save processed results as images, specify the template name of the output image file with `jpg` or `png` extension, for example: `-o output_%03d.jpg`. The actual file names are constructed from the template at runtime by replacing regular expression `%03d` with the frame number, resulting in the following: `output_000.jpg`, `output_001.jpg`, and so on.
To avoid disk space overrun in case of continuous input stream, like camera, you can limit the amount of data stored in the output file(s) with the `limit` option. The default value is 1000. To change it, you can apply the `-limit N` option, where `N` is the number of frames to store.

>**NOTE**: Windows\* systems may not have the Motion JPEG codec installed by default. If this is the case, you can download OpenCV FFMPEG back end using the PowerShell script provided with the OpenVINO &trade; install package and located at `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. The script should be run with administrative privileges if OpenVINO &trade; is installed in a system protected folder (this is a typical case). Alternatively, you can save results as images.

## Demo output

The demo uses OpenCV window to display the resulting video frame and detections.
The demo reports

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).
You can use both of these metrics to measure application-level performance.

## See also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
