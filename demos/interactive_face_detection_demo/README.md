# Interactive Face Detection Demo

This demo showcases Object Detection task applied for face recognition using sequence of neural networks.
Async API can improve overall frame-rate of the application, because rather than wait for inference to complete,
the application can continue operating on the host while accelerator is busy.
This demo executes four parallel infer requests for the Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, and Facial Landmarks Detection networks that run simultaneously. The corresponding pre-trained models are delivered with the product:
* `face-detection-adas-0001`, which is a primary detection network for finding faces
* `age-gender-recognition-retail-0013`, which is executed on top of the results of the first model and reports estimated age and gender for each detected face
* `head-pose-estimation-adas-0001`, which is executed on top of the results of the first model and reports estimated head pose in Tait-Bryan angles
* `emotions-recognition-retail-0003`, which is executed on top of the results of the first model and reports an emotion for each detected face
* `facial-landmarks-35-adas-0001`, which is executed on top of the results of the first model and reports normed coordinates of estimated facial landmarks

Other demo objectives are:

*	Video as input support via OpenCV\*
*	Visualization of the resulting face bounding boxes from Face Detection network
*	Visualization of age/gender, head pose, emotion information, and facial landmarks positions for each detected face

OpenCV is used to draw resulting bounding boxes, labels, and other information. You can copy and paste this code without pulling Inference Engine demo helpers into your application.

## How It Works

1.	The application reads command-line parameters and loads up to five networks depending on `-m...` options family to the Inference
Engine.
2.	The application gets a frame from the OpenCV VideoCapture.
3.	The application performs inference on the Face Detection network.
4.	The application performs four simultaneous inferences, using the Age/Gender, Head Pose, Emotions, and Facial Landmarks detection networks if they are specified in the command line.
5.	The application displays the results.

The new Async API operates with a new notion of the Infer Request that encapsulates the inputs/outputs and separates scheduling and waiting for result. For more information about Async API and the difference between Sync and Async modes performance, refer to **How it Works** and **Async API** sections in [Object Detection SSD, Async API Performance Showcase Demo](../object_detection_demo_ssd_async/README.md).


## Running

Running the application with the `-h` option yields the following usage message:

```sh
./interactive_face_detection_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

interactive_face_detection_demo [OPTION]
Options:

    -h                         Print a usage message
    -i "<path>"                Required. Path to a video file. Default value is "cam" to work with camera.
    -m "<path>"                Required. Path to an .xml file with a trained Face Detection model.
    -m_ag "<path>"             Optional. Path to an .xml file with a trained Age/Gender Recognition model.
    -m_hp "<path>"             Optional. Path to an .xml file with a trained Head Pose Estimation model.
    -m_em "<path>"             Optional. Path to an .xml file with a trained Emotions Recognition model.
    -m_lm "<path>"             Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.
      -l "<absolute_path>"     Required for CPU custom layers. Absolute path to a shared library with the kernels implementation.
          Or
      -c "<absolute_path>"     Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -d "<device>"              Target device for Face Detection network (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -d_ag "<device>"           Target device for Age/Gender Recognition network (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -d_hp "<device>"           Target device for Head Pose Estimation network (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -d_em "<device>"           Target device for Emotions Recognition network (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for a specified device.
    -d_lm "<device>"           Target device for Facial Landmarks Estimation network (CPU, GPU, FPGA, or MYRIAD). Demo will look for a suitable plugin for device specified.
    -n_ag "<num>"              Number of maximum simultaneously processed faces for Age/Gender Recognition network (default is 16)
    -n_hp "<num>"              Number of maximum simultaneously processed faces for Head Pose Estimation network (default is 16)
    -n_em "<num>"              Number of maximum simultaneously processed faces for Emotions Recognition network (default is 16)
    -n_lm "<num>"              Number of maximum simultaneously processed faces for Facial Landmarks Estimation network (default is 16)
    -dyn_ag                    Enable dynamic batch size for Age/Gender Recognition network
    -dyn_hp                    Enable dynamic batch size for Head Pose Estimation network
    -dyn_em                    Enable dynamic batch size for Emotions Recognition network
    -dyn_lm                    Enable dynamic batch size for Facial Landmarks Estimation network
    -async                     Enable asynchronous mode
    -no_wait                   Do not wait for key press in the end
    -no_show                   Do not show processed video
    -pc                        Enable per-layer performance report
    -r                         Output inference results as raw values
    -t                         Probability threshold for detections
```

Running the application with an empty list of options yields the usage message given above and an error message.

To run the demo, you can use public models or a set of pre-trained and optimized models:

* `open_model_zoo/intel_models/face-detection-adas-0001`
* `open_model_zoo/intel_models/age-gender-recognition-retail-0013`
* `open_model_zoo/intel_models/head-pose-estimation-adas-0001`
* `open_model_zoo/intel_models/emotions-recognition-retail-0003`
* `open_model_zoo/intel_models/facial-landmarks-35-adas-0001`

For example, to do inference on a GPU with the pre-trained models, run the following command:

```sh
./interactive_face_detection_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/face-detection-adas-0001.xml -m_ag <path_to_model>/age-gender-recognition-retail-0013.xml -m_hp <path_to_model>/head-pose-estimation-adas-0001.xml -m_em <path_to_model>/emotions-recognition-retail-0003.xml -m_lm <path_to_model>/facial-landmarks-35-adas-0001.xml -d GPU
```
> **NOTE**: Public models should be first converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer tool](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode, the demo reports:

* **OpenCV time**: frame decoding + time to render bounding boxes, labels, and display the results
* **Face Detection time**: inference time for the Face Detection network.

If Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, or Facial Landmarks Estimation networks are enabled, the additional information is reported:

* **Face Analysis Networks time**: combined inference time of simultaneously executed
Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, and Facial Landmarks Estimation networks.

## See Also
* [Using Inference Engine Demos](../Readme.md)
