# Interactive Face Detection C++ Demo

This demo showcases Object Detection task applied for face recognition using sequence of neural networks.
Async API can improve overall frame-rate of the application, because rather than wait for inference to complete,
the application can continue operating on the host while accelerator is busy.
This demo executes four parallel infer requests for the Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, and Facial Landmarks Detection networks that run simultaneously. You can use a set of the following pre-trained models with the demo:
* `face-detection-adas-0001`, which is a primary detection network for finding faces
* `age-gender-recognition-retail-0013`, which is executed on top of the results of the first model and reports estimated age and gender for each detected face
* `head-pose-estimation-adas-0001`, which is executed on top of the results of the first model and reports estimated head pose in Tait-Bryan angles
* `emotions-recognition-retail-0003`, which is executed on top of the results of the first model and reports an emotion for each detected face
* `facial-landmarks-35-adas-0002`, which is executed on top of the results of the first model and reports normed coordinates of estimated facial landmarks

For more information about the pre-trained models, refer to the [model documentation](../../models/intel/index.md).

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

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

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
    -i "<path>"                Required. Path to a video file (specify "cam" to work with camera).
    -o "<path>"                Optional. Path to an output video file.
    -m "<path>"                Required. Path to an .xml file with a trained Face Detection model.
    -m_ag "<path>"             Optional. Path to an .xml file with a trained Age/Gender Recognition model.
    -m_hp "<path>"             Optional. Path to an .xml file with a trained Head Pose Estimation model.
    -m_em "<path>"             Optional. Path to an .xml file with a trained Emotions Recognition model.
    -m_lm "<path>"             Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.
      -l "<absolute_path>"     Required for CPU custom layers. Absolute path to a shared library with the kernels implementation.
          Or
      -c "<absolute_path>"     Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
    -d "<device>"              Optional. Target device for Face Detection network (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -d_ag "<device>"           Optional. Target device for Age/Gender Recognition network (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -d_hp "<device>"           Optional. Target device for Head Pose Estimation network (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -d_em "<device>"           Optional. Target device for Emotions Recognition network (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -d_lm "<device>"           Optional. Target device for Facial Landmarks Estimation network (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -n_ag "<num>"              Optional. Number of maximum simultaneously processed faces for Age/Gender Recognition network (by default, it is 16)
    -n_hp "<num>"              Optional. Number of maximum simultaneously processed faces for Head Pose Estimation network (by default, it is 16)
    -n_em "<num>"              Optional. Number of maximum simultaneously processed faces for Emotions Recognition network (by default, it is 16)
    -n_lm "<num>"              Optional. Number of maximum simultaneously processed faces for Facial Landmarks Estimation network (by default, it is 16)
    -dyn_ag                    Optional. Enable dynamic batch size for Age/Gender Recognition network
    -dyn_hp                    Optional. Enable dynamic batch size for Head Pose Estimation network
    -dyn_em                    Optional. Enable dynamic batch size for Emotions Recognition network
    -dyn_lm                    Optional. Enable dynamic batch size for Facial Landmarks Estimation network
    -async                     Optional. Enable asynchronous mode
    -no_wait                   Optional. Do not wait for key press in the end.
    -no_show                   Optional. Do not show processed video.
    -pc                        Optional. Enable per-layer performance report
    -r                         Optional. Output inference results as raw values
    -t                         Optional. Probability threshold for detections
    -bb_enlarge_coef           Optional. Coefficient to enlarge/reduce the size of the bounding box around the detected face
    -dx_coef                   Optional. Coefficient to shift the bounding box around the detected face along the Ox axis
    -dy_coef                   Optional. Coefficient to shift the bounding box around the detected face along the Oy axis
    -fps                       Optional. Maximum FPS for playing video
    -loop_video                Optional. Enable playing video on a loop
    -no_smooth                 Optional. Do not smooth person attributes
    -no_show_emotion_bar       Optional. Do not show emotion bar
```

Running the application with an empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to do inference on a GPU with the OpenVINO&trade; toolkit pre-trained models, run the following command:

```sh
./interactive_face_detection_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/face-detection-adas-0001.xml -m_ag <path_to_model>/age-gender-recognition-retail-0013.xml -m_hp <path_to_model>/head-pose-estimation-adas-0001.xml -m_em <path_to_model>/emotions-recognition-retail-0003.xml -m_lm <path_to_model>/facial-landmarks-35-adas-0002.xml -d GPU
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
The demo reports total image throughput which includes frame decoding time, inference time, time to render bounding boxes and labels, and time to display the results.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo has been tested on the following Model Downloader available topologies: 
>* `age-gender-recognition-retail-0013`
>* `emotions-recognition-retail-0003`
>* `face-detection-adas-0001`
>* `facial-landmarks-35-adas-0002`
>* `head-pose-estimation-adas-0001`
> Other models may produce unexpected results on these devices.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
