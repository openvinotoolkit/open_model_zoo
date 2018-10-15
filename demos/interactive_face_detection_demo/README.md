# Interactive Face Detection Demo {#InferenceEngineInteractiveFaceDetectionDemoApplication}

This demo showcases Object Detection task applied for face recognition using sequence of neural networks.
Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete,
the application can continue operating on the host while accelerator is busy.
This demo maintains three parallel infer requests for the Age/Gender Recognition, Head Pose Estimation, and Emotions Recognition that run simultaneously.

Other demo objectives are:

*	Video as input support via OpenCV
*	Visualization of the resulting face bounding boxes from Face Detection network
*	Visualization of age/gender, head pose and emotion information for each detected face

OpenCV\* is used to draw resulting bounding boxes, labels, and other information. You can copy and paste this code without pulling Inference Engine demo helpers into your application

## How it Works

*	The application reads command line parameters and loads up to four networks depending on `-m...` options family to the Inference
Engine.
*	The application gets a frame from the OpenCV's VideoCapture.
*	The application performs inference on the frame detection network.
*	The application performs three simultaneous inferences, using the Age/Gender, Head Pose and Emotions detection networks if they are specified in command line.
*	The application displays the results.

The new Async API operates with a new notion of the Infer Request that encapsulates the inputs/outputs and separates scheduling and waiting for result. For more information about Async API and the difference between Sync and Async modes performance, refer to **How it Works** and **Async API** sections in [Object Detection SSD, Async API Performance Showcase Demo](@ref InferenceEngineObjectDetectionSSDDemoAsyncApplication).


## Running

Running the application with the `-h` option yields the following usage message:

```sh
./interactive_face_detection -h
InferenceEngine: 
    API version ............ <version>
    Build .................. <number>

interactive_face_detection [OPTION]
Options:

    -h                         Print a usage message.
    -i "<path>"                Optional. Path to an video file. Default value is "cam" to work with camera.
    -m "<path>"                Required. Path to an .xml file with a trained face detection model.
    -m_ag "<path>"             Optional. Path to an .xml file with a trained age gender model.
    -m_hp "<path>"             Optional. Path to an .xml file with a trained head pose model.
    -m_em "<path>"             Optional. Path to an .xml file with a trained emotions model.
      -l "<absolute_path>"     Required for MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"     Required for clDNN (GPU)-targeted custom kernels.Absolute path to the xml file with the kernels desc.
    -d "<device>"              Specify the target device for Face Detection (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -d_ag "<device>"           Specify the target device for Age Gender Detection (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -d_hp "<device>"           Specify the target device for Head Pose Detection (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -d_em "<device>"           Specify the target device for Emotions Detection (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -n_ag "<num>"              Specify number of maximum simultaneously processed faces for Age Gender Detection (default is 16).
    -n_hp "<num>"              Specify number of maximum simultaneously processed faces for Head Pose Detection (default is 16).
    -n_em "<num>"              Specify number of maximum simultaneously processed faces for Emotions Detection (default is 16).
    -no_wait                   No wait for key press in the end.
    -no_show                   No show processed video.
    -pc                        Enables per-layer performance report.
    -r                         Inference results as raw values.
    -t                         Probability threshold for detections.

```

Running the application with an empty list of options yields the usage message given above and an error message.

To run the demo, you can use public models or a set of pre-trained and optimized models delivered with the package:

* `<INSTAL_DIR>/deployment_tools/intel_models/face-detection-adas-0001`
* `<INSTAL_DIR>/deployment_tools/intel_models/age-gender-recognition-retail-0013`
* `<INSTAL_DIR>/deployment_tools/intel_models/head-pose-estimation-adas-0001`
* `<INSTAL_DIR>/deployment_tools/intel_models/emotions-recognition-retail-0003`

For example, to do inference on a GPU with the OpenVINO&trade; toolkit pre-trained models, run the following command:

```sh
./interactive_face_detection -i <path_to_video>/inputVideo.mp4 -m face-detection-adas-0001.xml -m_ag age-gender-recognition-retail-0013.xml -m_hp head-pose-estimation-adas-0001.xml -m_em emotions-recognition-retail-0003.xml -d GPU
```
**NOTE**: Public models should be first converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer](@ref MODevGuide) tool.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode the demo reports:

* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and displaying the results.
* **Face Detection time**: inference time for the face Detection network. 

If Age/Gender recognition, Head Pose estimation, or Emotions recognition are enabled, the additional information is reported:

* **Age Gender + Head Pose + Emotions Detection time**: combined inference time of simultaneously executed
age/gender, head pose, and emotion recognition networks.

## See Also
* [Using Inference Engine Demos](@ref DemosOverview)
