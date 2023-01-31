# Face Recognition C++ Demo

![](./face_recognition_demo.gif)

This example demonstrates an approach to create interactive applications
for video processing. It shows the basic architecture for building model
pipelines supporting model placement on different devices and simultaneous
parallel or sequential execution using OpenVINO library.
In particular, this demo uses 4 models to build a pipeline able to detect
faces on videos, their keypoints (aka "landmarks"),
recognize persons using the provided faces database (the gallery) and estimate probabilities
whether spoof or real persons on video or image.
The following pretrained models can be used:

* `face-detection-retail-0004` and `face-detection-adas-0001`, to detect faces and predict their bounding boxes;
* `landmarks-regression-retail-0009`, to predict face keypoints;
* `face-reidentification-retail-0095`, `Sphereface`, `facenet-20180408-102900` or `face-recognition-resnet100-arcface-onnx` to recognize persons.
* `anti-spoof-mn3`, which is executed on top of the results of the detection model and reports estimated probability whether spoof or real face is shown

## How it works

The application is invoked from command line. It reads the specified input
video stream frame-by-frame, be it a camera device or a video file,
and performs independent analysis of each frame. In order to make predictions
the application deploys 4 models on the specified devices using OpenVINO
library and runs them in asynchronous manner.
There are 3 user modes for this demo application:
1. Only face detetion. In this mode, an input frame is processed by the face detection model to predict face bounding boxes.
  To do face detection, use only `-mfd` flag. Example:
  ```
  ./face_recognition_demo \
     -i <path_to_video>/input_video.mp4 \
     -mfd <path_to_model>/face-detection-retail-0004.xml
  ```
2. Face Recognition mode. In this case after face detection, face keypoints
   are predicted by the Landmarks model and as final step face recognition model uses keypoints to align faces
   and match found faces with faces from face gallery, which should be defined by user.
   So, in this mode user should provide 3 flags `-mfd`, `-mlm`, `-mreid`. Example:
   ```
   ./face_recognition_demo
     -i <path_to_video>/input_video.mp4
     -mfd <path_to_model>/face-detection-retail-0004.xml
     -mlm <path_to_model>/landmarks-regression-retail-0009.xml
     -mreid <path_to_model>/face-reidentification-retail-0095.xml
     -fg "/home/face_gallery"
   ```
3. With Anti-Spoof model. In this case 4 models are working and for all recognized faces demo applies anti-spoof model,
   which estimate probability whether spoof or real faces on video.
   So, in this mode user should provide 3 flags `-mfd`, `-mlm`, `-mreid`. Example:
   ```
   ./face_recognition_demo
     -i <path_to_video>/input_video.mp4
     -mfd <path_to_model>/face-detection-retail-0004.xml
     -mlm <path_to_model>/landmarks-regression-retail-0009.xml
     -mreid <path_to_model>/face-reidentification-retail-0095.xml
     -mas <path_to_model>/anti-spoof-mn3.xml
     -fg "/home/face_gallery"
   ```

After all computations the processing results are
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
* Sphereface
* anti-spoof-mn3

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
The application can build gallery while working, that is
controlled by `--allow_grow` flag. In that mode the user will
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

Running the application with the `-h` option:

```
        [ -h]                                                show the help message and exit
        [--help]                                             print help on all arguments
        [ -i <INPUT>]                                        an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0
         -mfd <MODEL FILE>                                  path to the Face Detection model (.xml) file.
        [-mlm <MODEL FILE>]                                 path to the Facial Landmarks Regression Retail model (.xml) file
        [-mreid <MODEL FILE>]                               path to the Face Recognition model (.xml) file.
        [-mas <MODEL FILE>]                                 path to the Antispoofing Classification model (.xml) file.
        [ -t_fd <NUMBER>]                                    probability threshold for face detections. Default is 0.5
        [ --input_shape <STRING>]                            specify the input shape for detection network in (width x height) format. Input of model will be reshaped according specified shape.Example: 1280x720. Shape of network input used by default.
        [ -t_reid <NUMBER>]                                  cosine distance threshold between two vectors for face reidentification. Default is 0.7
        [ -exp <NUMBER>]                                     expand ratio for bbox before face recognition. Default is 1.0
        [--greedy_reid_matching] ([--nogreedy_reid_matching])(don't) use faster greedy matching algorithm in face reid.
        [-fg <GALLERY PATH>]                                 path to a faces gallery directory.
        [--allow_grow] ([--noallow_grow])                    (dont't) allow to grow faces gallery and to dump on disk.
        [--crop_gallery] ([--nocrop_gallery])                (dont't) crop images during faces gallery creation.
        [ -dfd <DEVICE>]                                    specify a device Face Detection model to infer on (the list of available devices is shown below). Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU
        [ -dlm <DEVICE>]                                    specify a device for Landmarks Regression model to infer on (the list of available devices is shown below). Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU
        [ -dreid <DEVICE>]                                  specify a target device for Face Reidentification model to infer on (the list of available devices is shown below). Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU
        [ -das <DEVICE>]                                    specify a device for Anti-spoofing model to infer on (the list of available devices is shown below). Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU
        [--lim <NUMBER>]                                     number of frames to store in output. If 0 is set, all frames are stored. Default is 1000
        [ -o <OUTPUT>]                                       name of the output file(s) to save.
        [--loop]                                             enable reading the input in a loop
        [--nthreads <integer>]                               number of threads for TFLite model.
        [--show] ([--noshow])                                (don't) show output
        [ -u <DEVICE>]                                       resource utilization graphs. Default is cdm. c - average CPU load, d - load distribution over cores, m - memory usage, h - hide
        Key bindings:
                Q, q, Esc - Quit
                P, p, 0, spacebar - Pause
                C - average CPU load, D - load distribution over cores, M - memory usage, H - hide
```

Example of a valid command line to run the application:

``` sh

./face_recognition_demo \
  -i <path_to_video>/input_video.mp4 \
  -m_fd <path_to_model>/face-detection-retail-0004.xml \
  -m_lm <path_to_model>/landmarks-regression-retail-0009.xml \
  -m_reid <path_to_model>/face-reidentification-retail-0095.xml \
  --verbose \
  -fg "/home/face_gallery"
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
* [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
