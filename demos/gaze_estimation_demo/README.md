# Gaze Estimation Demo

This demo showcases the work of gaze estimation model.
The corresponding pre-trained model `gaze-estimation-adas-0002` is delivered with the product.

The demo also relies on the following auxiliary networks:
* `face-detection-retail-0004` or `face-detection-adas-0001` detection networks for finding faces
* `head-pose-estimation-adas-0001`, which estimates head pose in Tait-Bryan angles, serving as an input for gaze estimation model
* `facial-landmarks-35-adas-0002`, which estimates coordinates of facial landmarks for detected faces. The keypoints at the corners of eyes are used to locate eyes regions required for the gaze estimation model

For more information about the pre-trained models, refer to the [model documentation](../../models/intel/index.md).

Other demo objectives are:
* Video/Camera as inputs, via OpenCV*
* Visualization of gaze estimation results, and, optionally, results of inference on auxiliary models

## How It Works

1. The application reads command-line parameters and loads four networks to the Inference Engine
2. The application gets a frame from the OpenCV VideoCapture
3. The application performs inference on auxiliary models to obtain head pose angles and images of eyes regions serving as an input for gaze estimation model
4. The application performs inference on gaze estimation model using inference results of auxiliary models
5. The application shows the results

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./gaze_estimation_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

gaze_estimation_demo [OPTION]
Options:

    -h                       Print a usage message.
    -i "<path>"              Optional. Path to a video file. Default value is "cam" to work with camera.
    -m "<path>"              Required. Path to an .xml file with a trained Gaze Estimation model.
    -m_fd "<path>"           Required. Path to an .xml file with a trained Face Detection model.
    -m_hp "<path>"           Required. Path to an .xml file with a trained Head Pose Estimation model.
    -m_lm "<path>"           Required. Path to an .xml file with a trained Facial Landmarks Estimation model.
    -d "<device>"            Optional. Target device for Gaze Estimation network (the list of available devices is shown below). Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device. Default value is "CPU".
    -d_fd "<device>"         Optional. Target device for Face Detection network (the list of available devices is shown below). Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device. Default value is "CPU".
    -d_hp "<device>"         Optional. Target device for Head Pose Estimation network (the list of available devices is shown below). Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device. Default value is "CPU".
    -d_lm "<device>"         Optional. Target device for Facial Landmarks Estimation network (the list of available devices is shown below). Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device. Default value is "CPU".
    -res "<WxH>"             Optional. Set camera resolution in format WxH.
    -fd_reshape              Optional. Reshape Face Detector network so that its input resolution has the same aspect ratio as the input frame.
    -no_show                 Optional. Do not show processed video.
    -pc                      Optional. Enable per-layer performance report.
    -r                       Optional. Output inference results as raw values.
    -t                       Optional. Probability threshold for Face Detector. The default value is 0.5.
```

Running the application with an empty list of options yields an error message.

To run the demo, you can use public or pre-trained and optimized `gaze-estimation-adas-0002` model, and the auxiliary models. To download the pre-trained models, use the OpenVINO Model Downloader or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with another trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to do inference on a CPU, run the following command:

```sh
./gaze_estimation_demo -i <path_to_video>/input_video.mp4  -m <path_to_model>/gaze-estimation-adas-0002.xml -m_fd <path_to_model>/face-detection-retail-0004.xml -m_hp <path_to_model>/head-pose-estimation-adas-0001.xml -m_lm <path_to_model>/facial-landmarks-35-adas-0002.xml
```

## Demo Output

The demo uses OpenCV to display the resulting frame with marked gaze vectors, text reports of **FPS** (frames per second performance) for the demo, and, optionally, marked facial landmarks, head pose angles, and face bounding boxes. 
By default, it shows only gaze estimation results. To see inference results of auxiliary networks, use run-time control keys.

### Run-Time Control Keys

The demo allows you to control what information is displayed in run-time.
The following keys are supported:
* G - to toggle displaying gaze vector
* D - to toggle displaying face detector bounding boxes 
* H - to toggle displaying head pose information
* L - to toggle displaying facial landmarks 
* A - to switch on displaying all inference results
* N - to switch off displaying all inference results
* F - to flip frames horizontally
* Esc - to quit the demo

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo has been tested on the following Model Downloader available topologies: 
>* `face-detection-adas-0001`
>* `face-detection-retail-0004`
>* `facial-landmarks-35-adas-0002`
>* `gaze-estimation-adas-0002`
>* `head-pose-estimation-adas-0001`
> Other models may produce unexpected results on these devices.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
