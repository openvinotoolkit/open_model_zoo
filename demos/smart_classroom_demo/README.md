# Smart Classroom C++ Demo

The demo shows an example of joint usage of several neural networks to detect three basic actions (sitting, standing, raising hand) and recognize people by faces in the classroom environment. The demo uses Async API for action and face detection networks. It allows to parallelize execution of face recognition and detection: while face recognition is running on one accelerator, face and action detection could be performed on another. You can use a set of the following pre-trained models with the demo:

* `face-detection-adas-0001`, which is a primary detection network for finding faces.
* `landmarks-regression-retail-0009`, which is executed on top of the results from the first network and outputs
a vector of facial landmarks for each detected face.
* `face-reidentification-retail-0095`,  which is executed on top of the results from the first network and outputs
a vector of features for each detected face.
* `person-detection-action-recognition-0005`, which is a detection network for finding persons and simultaneously predicting their current actions.
* `person-detection-raisinghand-recognition-0001`, which is a detection network for finding students and simultaneously predicting their current actions (in contrast with the previous model, predicts only if a student raising hand or not).
* `person-detection-action-recognition-teacher-0002`, which is a detection network for finding persons and simultaneously predicting their current actions.

For more information about the pre-trained models, refer to the [Open Model Zoo](https://github.com/opencv/open_model_zoo/blob/master/intel_models/index.md) repository on GitHub*.

## How It Works

On the start-up, the application reads command line parameters and loads four networks to the Inference Engine for execution on different devices depending on `-m...` options family. Upon getting a frame from the OpenCV VideoCapture, it performs inference of Face Detection and Action Detection networks. After that, the ROIs obtained by Face Detector are fed to the Facial Landmarks Regression network. Then landmarks are used to align faces by affine transform and feed them to the Face Recognition network. The recognized faces are matched with detected actions to find an action for a recognized person for each frame.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified.

## Creating a Gallery for Face Recognition

To recognize faces on a frame, the demo needs a gallery of reference images. Each image should contain a tight crop of face. You can create the gallery from an arbitrary list of images:
1. Put images containing tight crops of frontal-oriented faces to a separate empty folder. Each identity could have multiple images. Name images as `id_name.0.png, id_name.1.png, ...`.
2. Run the `create_list.py <path_to_folder_with_images>` command to get a list of files and identities in `.json` format.

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./smart_classroom_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

smart_classroom_demo [OPTION]
Options:

    -h                             Print a usage message.
    -i '<path>'                    Required. Path to a video or image file. Default value is "cam" to work with camera.
    -m_act '<path>'                Required. Path to the Person/Action Detection Retail model (.xml) file.
    -m_fd '<path>'                 Required. Path to the Face Detection Retail model (.xml) file.
    -m_lm '<path>'                 Required. Path to the Facial Landmarks Regression Retail model (.xml) file.
    -m_reid '<path>'               Required. Path to the Face Reidentification Retail model (.xml) file.
    -l '<absolute_path>'           Optional. For CPU custom layers, if any. Absolute path to a shared library with the kernels implementation.
          Or
    -c '<absolute_path>'           Optional. For GPU custom kernels, if any. Absolute path to an .xml file with the kernels description.
    -d_act '<device>'              Optional. Specify the target device for Person/Action Detection Retail (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO).
    -d_fd '<device>'               Optional. Specify the target device for Face Detection Retail (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO).
    -d_lm '<device>'               Optional. Specify the target device for Landmarks Regression Retail (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO).
    -d_reid '<device>'             Optional. Specify the target device for Face Reidentification Retail (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO).
    -out_v  '<path>'               Optional. File to write output video with visualization to.
    -pc                            Optional. Enables per-layer performance statistics.
    -r                             Optional. Output Inference results as raw values.
    -ad                            Optional. Output file name to save per-person action statistics in.
    -t_ad                          Optional. Probability threshold for person/action detection.
    -t_ar                          Optional. Probability threshold for action recognition.
    -t_fd                          Optional. Probability threshold for face detections.
    -inh_fd                        Optional. Input image height for face detector.
    -inw_fd                        Optional. Input image width for face detector.
    -exp_r_fd                      Optional. Expand ratio for bbox before face recognition.
    -t_reid                        Optional. Cosine distance threshold between two vectors for face reidentification.
    -fg                            Optional. Path to a faces gallery in .json format.
    -no_show                       Optional. Do not show processed video.
    -last_frame                    Optional. Last frame number to handle in demo. If negative, handle all input video.
    -teacher_id                    Optional. ID of a teacher. You must also set a faces gallery parameter (-fg) to use it.
    -min_ad                        Optional. Minimum action duration in seconds.
    -d_ad                          Optional. Maximum time difference between actions in seconds.
    -student_ac                    Optional. List of student actions separated by a comma.
    -teacher_ac                    Optional. List of teacher actions separated by a comma.
    -a_id                          Optional. Target action name.
    -a_top                         Optional. Number of first K students. If this parameter is positive, the demo detects first K persons with the action, pointed by the parameter "a_id"
    -crop_gallery                  Optional. Crop images during faces gallery creation.
    -t_reg_fd                      Optional. Probability threshold for face detections during database registration.
    -min_size_fr                   Optional. Minimum input size for faces during database registration.
    -al                            Optional. Output file name to save per-person action detections in.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).

Example of a valid command line to run the application with pre-trained models for recognizing students actions:
```sh
./smart_classroom_demo -m_act <path_to_model>/person-detection-action-recognition-0005.xml \
                       -m_fd <path_to_model>/face-detection-adas-0001.xml \
                       -m_reid <path_to_model>/face-reidentification-retail-0095.xml \
                       -m_lm <path_to_model>/landmarks-regression-retail-0009.xml \
                       -fg <path_to_faces_gallery.json> \
                       -i <path_to_video>
```
> **NOTE**: To recognize actions of students, use `person-detection-action-recognition-0005` model.

Example of a valid command line to run the application for recognizing actions of a teacher:
```sh
./smart_classroom_demo -m_act <path_to_model>/person-detection-action-recognition-teacher-0002.xml \
                       -m_fd <path_to_model>/face-detection-adas-0001.xml \
                       -m_reid <path_to_model>/face-reidentification-retail-0095.xml \
                       -m_lm <path_to_model>/landmarks-regression-retail-0009.xml \
                       -fg <path to faces_gallery.json> \
                       -teacher_id <ID of a teacher in the face gallery> \
                       -i <path_to_video>
```
> **NOTE**: To recognize actions of a teacher, use `person-detection-action-recognition-teacher-0002` model.

Example of a valid command line to run the application for recognizing first raised-hand students:
```sh
./smart_classroom_demo -m_act <path_to_model>/person-detection-raisinghand-recognition-0001.xml \
                       -a_top <number of first raised-hand students> \
                       -i <path_to_video>
```
> **NOTE**: To recognize raising hand action of students, use `person-detection-raisinghand-recognition-0001` model.

## Demo Output

The demo uses OpenCV to display the resulting frame with labeled actions and faces.

## See Also
* [Using Inference Engine Demos](../Readme.md)
* [Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
