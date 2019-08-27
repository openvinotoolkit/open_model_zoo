# Interactive Face Recognition Demo

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

For more information about the pre-trained models, refer to the [model documentation](../../../models/intel/index.md).

### How it works

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

### Installation and dependencies

The demo depends on:
- OpenVINO library (2018R5 or newer)
- Python (any of 2.7+ or 3.4+, which is supported by OpenVINO)
- OpenCV (>=3.4.0)

To install all the required Python modules you can use:

``` sh
pip install -r requirements.txt
```

### Running the demo:

Running the application with the `-h` option or without
any arguments yields the following message:

``` sh
python ./face_recognition_demo.py -h

usage: face_recognition_demo.py [-h] [-i PATH] [-o PATH] [--no_show] [-tl]
                                [-cw CROP_WIDTH] [-ch CROP_HEIGHT] -fg PATH
                                [--run_detector] -m_fd PATH -m_lm PATH -m_reid
                                PATH [-d_fd {CPU,GPU,FPGA,MYRIAD,HETERO}]
                                [-d_lm {CPU,GPU,FPGA,MYRIAD,HETERO}]
                                [-d_reid {CPU,GPU,FPGA,MYRIAD,HETERO}]
                                [-l PATH] [-c PATH] [-v] [-pc] [-t_fd [0..1]]
                                [-t_id [0..1]] [-exp_r_fd NUMBER]

optional arguments:
  -h, --help            show this help message and exit

General:
  -i PATH, --input PATH
                        (optional) Path to the input video ('cam' for the
                        camera, default)
  -o PATH, --output PATH
                        (optional) Path to save the output video to
  --no_show             (optional) Do not display output
  -tl, --timelapse      (optional) Auto-pause after each frame
  -cw CROP_WIDTH, --crop_width CROP_WIDTH
                        (optional) Crop the input stream to this width
                        (default: no crop). Both -cw and -ch parameters should
                        be specified to use crop.
  -ch CROP_HEIGHT, --crop_height CROP_HEIGHT
                        (optional) Crop the input stream to this height
                        (default: no crop). Both -cw and -ch parameters should
                        be specified to use crop.

Faces database:
  -fg PATH              Path to the face images directory
  --run_detector        (optional) Use Face Detection model to find faces on
                        the face images, otherwise use full images.
  --allow_grow          (optional) Flag to allow growing the face database,
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
  -m_fd PATH            Path to the Face Detection model XML file
  -m_lm PATH            Path to the Facial Landmarks Regression model XML file
  -m_reid PATH          Path to the Face Reidentification model XML file

Inference options:
  -d_fd {CPU,GPU,FPGA,MYRIAD,HETERO}
                        (optional) Target device for the Face Detection model
                        (default: CPU)
  -d_lm {CPU,GPU,FPGA,MYRIAD,HETERO}
                        (optional) Target device for the Facial Landmarks
                        Regression model (default: CPU)
  -d_reid {CPU,GPU,FPGA,MYRIAD,HETERO}
                        (optional) Target device for the Face Reidentification
                        model (default: CPU)
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
                        vectors for face identification (default: 0.3)
  -exp_r_fd NUMBER      (optional) Scaling ratio for bboxes passed to face
                        recognition (default: 1.15)
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
-l <path_to_cpu_extensions>/libcpu_extension_sse4.so \
--verbose \
-fg "/home/face_gallery"
```

Windows (`cmd`, `powershell`) (assuming OpenVINO installed in `C:/Intel/openvino`):

``` powershell
# Set up the environment
call C:/Intel/openvino/bin/setupvars.bat

python ./face_recognition_demo.py ^
-m_fd <path_to_model>/face-detection-retail-0004.xml ^
-m_lm <path_to_model>/landmarks-regression-retail-0009.xml ^
-m_reid <path_to_model>/face-reidentification-retail-0095.xml ^
-l <path_to_cpu_extensions>/cpu_extension_avx2.dll ^
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
* [Using Inference Engine Demos](../../README.md)
