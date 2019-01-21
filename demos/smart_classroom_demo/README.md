# Smart Classroom Demo

The demo shows an example of joint usage of several neural networks to detect three basic actions (sitting, standing, raising hand) and recognize people by faces in the classroom environment. The demo uses Async API for action and face detection networks. It allows to parallelize execution of face recognition and detection: while face recognition is running on one accelerator, face and action detection could be performed on another. The corresponding pre-trained models are delivered with the product:

* `face-detection-adas-0001`, which is a primary detection network for finding faces.
* `landmarks-regression-retail-0009`, which is executed on top of the results from the first network and outputs
a vector of facial landmarks for each detected face.
* `face-reidentification-retail-0095`,  which is executed on top of the results from the first network and outputs
a vector of features for each detected face.
* `person-detection-action-recognition-0003`, which is a detection network for finding persons and simultaneously predicting their current actions.

## How It Works

On the start-up, the application reads command line parameters and loads four networks to the Inference Engine for execution on different devices depending on `-d...` options family. Upon getting a frame from the OpenCV VideoCapture, it performs inference of Face Detection and Action Detection networks. After that, the ROIs obtained by Face Detector are fed to the Facial Landmarks Regression network. Then landmarks are used to align faces by affine transform and feed them to the Face Recognition network. The recognized faces are matched with detected actions to find an action for a recognized person for each frame.

## Creating a Gallery for Face Recognition

To recognize faces on a frame, the demo needs a gallery of reference images. Each image should contain a tight crop of face. You can create the gallery from an arbitrary list of images:
1. Put images containing tight crops of frontal-oriented faces to a separate empty folder. Each identity could have multiple images. Name images as `id_name.0.png, id_name.1.png, ...`.
2. Run the `create_list.py <path_to_folder_with_images>` command to get a list of files and identities in `.json` format.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
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
    -d_act '<device>'              Optional. Specify the target device for Person/Action Detection Retail (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -d_fd '<device>'               Optional. Specify the target device for Face Detection Retail (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -d_lm '<device>'               Optional. Specify the target device for Landmarks Regression Retail (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -d_reid '<device>'             Optional. Specify the target device for Face Reidentification Retail (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -out_v  '<path>'               Optional. File to write output video with visualization to.
    -pc                            Optional. Enables per-layer performance statistics.
    -r                             Optional. Output Inference results as raw values.
    -ad                            Optional. Output file name to save per-person action statistics in.
    -t_act                         Optional. Probability threshold for persons/actions detections.
    -t_fd                          Optional. Probability threshold for face detections.
    -inh_fd                        Optional. Input image height for face detector.
    -inw_fd                        Optional. Input image width for face detector.
    -exp_r_fd                      Optional. Expand ratio for bbox before face recognition.
    -t_reid                        Optional. Cosine distance threshold between two vectors for face reidentification.
    -fg                            Optional. Path to a faces gallery in .json format.
    -no_show                       Optional. Do not show processed video.
    -last_frame                    Optional. Last frame number to handle in demo. If negative, handle all input video.
```

Running the application with the empty list of options yields the usage message given above and an error message.

Example of a valid command line to run the application:
```sh
./smart_classroom_demo -m_act <path to the person/action detection retail model .xml file> \
                       -m_fd <path to the face detection retail model .xml file> \
                       -m_reid <path to the face reidentification retail model .xml file> \
                       -m_lm <path to the landmarks regression retail model .xml file> \
                       -fg <path to faces_gallery.json> \
                       -i <path to the input video>
```

> **NOTE**: Public Caffe\* models should be first converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer tool](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).

## Demo Output

The demo uses OpenCV to display the resulting frame with labeled actions and faces.

## See Also
* [Using Inference Engine Demos](../Readme.md)
