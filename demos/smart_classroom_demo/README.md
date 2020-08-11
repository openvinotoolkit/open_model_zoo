# Smart Classroom C++ Demo

The demo shows an example of joint usage of several neural networks to detect student actions (sitting, standing, raising hand for the `person-detection-action-recognition-0005` model and sitting, writing, raising hand, standing, turned around, lie on the desk for the `person-detection-action-recognition-0006` model) and recognize people by faces in the classroom environment. The demo uses Async API for action and face detection networks. It allows to parallelize execution of face recognition and detection: while face recognition is running on one accelerator, face and action detection could be performed on another. You can use a set of the following pre-trained models with the demo:

* `face-detection-adas-0001`, which is a primary detection network for finding faces.
* `landmarks-regression-retail-0009`, which is executed on top of the results from the first network and outputs
a vector of facial landmarks for each detected face.
* `person-detection-action-recognition-0005`, which is a detection network for finding persons and simultaneously predicting their current actions (3 actions - sitting, standing, raising hand).
* `person-detection-action-recognition-0006`, which is a detection network for finding persons and simultaneously predicting their current actions (6 actions: sitting, writing, raising hand, standing, turned around, lie on the desk).
* `person-detection-raisinghand-recognition-0001`, which is a detection network for finding students and simultaneously predicting their current actions (in contrast with the previous model, predicts only if a student raising hand or not).
* `person-detection-action-recognition-teacher-0002`, which is a detection network for finding persons and simultaneously predicting their current actions.

For more information about the pre-trained models, refer to the [model documentation](../../models/intel/index.md).

## How It Works

On the start-up, the application reads command line parameters and loads four networks to the Inference Engine for execution on different devices depending on `-m...` options family. Upon getting a frame from the OpenCV VideoCapture, it performs inference of Face Detection and Action Detection networks. After that, the ROIs obtained by Face Detector are fed to the Facial Landmarks Regression network. Then landmarks are used to align faces by affine transform and feed them to the Face Recognition network. The recognized faces are matched with detected actions to find an action for a recognized person for each frame.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Creating a Gallery for Face Recognition

To recognize faces on a frame, the demo needs a gallery of reference images. Each image should contain a tight crop of face. You can create the gallery from an arbitrary list of images:
1. Put images containing tight crops of frontal-oriented faces (or use `-crop_gallery` key for the demo) to a separate empty folder. Each identity must have only one image. Name images as `id_name0.png, id_name1.png, ...`.
2. Run the `create_list.py <path_to_folder_with_images>` command to get a list of files and identities in `.json` format.

## Running

Running the application with the `-h` option yields the following usage message:
```
./smart_classroom_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

smart_classroom_demo [OPTION]
Options:

    -h                             Print a usage message.
    -i                             Required. An input to process. The input must be a single image, a folder of images or anything that cv::VideoCapture can process.
    -loop                          Optional. Enable reading the input in a loop.
    -limit                         Optional. Read length limit before stopping or restarting reading the input.
    -m_act '<path>'                Required. Path to the Person/Action Detection Retail model (.xml) file.
    -m_fd '<path>'                 Required. Path to the Face Detection model (.xml) file.
    -m_lm '<path>'                 Required. Path to the Facial Landmarks Regression Retail model (.xml) file.
    -m_reid '<path>'               Required. Path to the Face Reidentification Retail model (.xml) file.
    -l '<absolute_path>'           Optional. For CPU custom layers, if any. Absolute path to a shared library with the kernels implementation.
          Or
    -c '<absolute_path>'           Optional. For GPU custom kernels, if any. Absolute path to an .xml file with the kernels description.
    -d_act '<device>'              Optional. Specify the target device for Person/Action Detection Retail (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_fd '<device>'               Optional. Specify the target device for Face Detection Retail (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_lm '<device>'               Optional. Specify the target device for Landmarks Regression Retail (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_reid '<device>'             Optional. Specify the target device for Face Reidentification Retail (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -out_v  '<path>'               Optional. File to write output video with visualization to.
    -greedy_reid_matching          Optional. Use faster greedy matching algorithm in face reid.
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
    -teacher_id                    Optional. ID of a teacher. You must also set a faces gallery parameter (-fg) to use it.
    -no_show                       Optional. Do not show processed video.
    -min_ad                        Optional. Minimum action duration in seconds.
    -d_ad                          Optional. Maximum time difference between actions in seconds.
    -student_ac                    Optional. List of student actions separated by a comma.
    -top_ac                        Optional. List of student actions (for top-k mode) separated by a comma.
    -teacher_ac                    Optional. List of teacher actions separated by a comma.
    -top_id                        Optional. Target action name.
    -a_top                         Optional. Number of first K students. If this parameter is positive, the demo detects first K persons with the action, pointed by the parameter 'top_id'
    -crop_gallery                  Optional. Crop images during faces gallery creation.
    -t_reg_fd                      Optional. Probability threshold for face detections during database registration.
    -min_size_fr                   Optional. Minimum input size for faces during database registration.
    -al                            Optional. Output file name to save per-person action detections in.
    -ss_t                          Optional. Number of frames to smooth actions.
    -u                             Optional. List of monitors to show initially.
```

Running the application with the empty list of options yields an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

Example of a valid command line to run the application with pre-trained models for recognizing students actions:
```sh
./smart_classroom_demo -m_act <path_to_model>/person-detection-action-recognition-0005.xml \
                       -m_fd <path_to_model>/face-detection-adas-0001.xml \
                       -i <path_to_video>
```
> **NOTE**: To recognize actions of students, use `person-detection-action-recognition-0005` model for 3 basic actions and `person-detection-action-recognition-0006` model for 6 actions.

Example of a valid command line to run the application for recognizing first raised-hand students:
```sh
./smart_classroom_demo -m_act <path_to_model>/person-detection-raisinghand-recognition-0001.xml \
                       -a_top <number of first raised-hand students> \
                       -i <path_to_video>
```
> **NOTE**: To recognize raising hand action of students, use `person-detection-raisinghand-recognition-0001` model.

## Demo Output

The demo uses OpenCV to display the resulting frame with labeled actions and faces.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo has been tested on the following Model Downloader available topologies:
>* `face-detection-adas-0001`
>* `landmarks-regression-retail-0009`
>* `person-detection-action-recognition-0005`
> Other models may produce unexpected results on these devices.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
