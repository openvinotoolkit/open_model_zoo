# Crossroad Camera C++ Demo

This demo provides an inference pipeline for persons' detection, recognition and reidentification. The demo uses Person Detection network followed by the Person Attributes Recognition and Person Reidentification Retail networks applied on top of the detection results. You can use a set of the following pre-trained models with the demo:

* `person-vehicle-bike-detection-crossroad-0078`, which is a primary detection network for finding the persons (and other objects if needed)
* `person-attributes-recognition-crossroad-0230`, which is executed on top of the results from the first network and
reports person attributes like gender, has hat, has long-sleeved clothes
* `person-reidentification-retail-0079`, which is executed on top of the results from the first network and prints
a vector of features for each detected person. This vector is used to conclude if it is already detected person or not.

For more information about the pre-trained models, refer to the [model documentation](../../models/intel/index.md).

Other demo objectives are:
* Images/Video/Camera as inputs, via OpenCV*
* Example of simple networks pipelining: Person Attributes and Person Reidentification networks are executed on top of
the Person Detection results
* Visualization of Person Attributes and Person Reidentification (REID) information for each detected person


## How It Works

On the start-up, the application reads command line parameters and loads the specified networks. The Person Detection
network is required, the other two are optional.

Upon getting a frame from the OpenCV VideoCapture, the application performs inference of Person Detection network, then performs another
two inferences of Person Attributes Recognition and Person Reidentification Retail networks if they were specified in the
command line, and displays the results.

In case of a Person Reidentification Retail network specified, the resulting vector is generated for each detected person. This vector is
compared one-by-one with all previously detected persons vectors using cosine similarity algorithm. If comparison result
is greater than the specified (or default) threshold value, it is concluded that the person was already detected and a known
REID value is assigned. Otherwise, the vector is added to a global list, and new REID value is assigned.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./crossroad_camera_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

crossroad_camera_demo [OPTION]
Options:

    -h                           Print a usage message.
    -i "<path>"                  Required. Path to a video or image file. Default value is "cam" to work with camera.
    -m "<path>"                  Required. Path to the Person/Vehicle/Bike Detection Crossroad model (.xml) file.
    -m_pa "<path>"               Optional. Path to the Person Attributes Recognition Crossroad model (.xml) file.
    -m_reid "<path>"             Optional. Path to the Person Reidentification Retail model (.xml) file.
      -l "<absolute_path>"       Optional. For CPU custom layers, if any. Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"       Optional. For GPU custom kernels, if any. Absolute path to the xml file with the kernels desc.
    -d "<device>"                Optional. Specify the target device for Person/Vehicle/Bike Detection. The list of available devices is shown below. Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_pa "<device>"             Optional. Specify the target device for Person Attributes Recognition. The list of available devices is shown below. Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_reid "<device>"           Optional. Specify the target device for Person Reidentification Retail. The list of available devices is shown below. Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -pc                          Optional. Enables per-layer performance statistics.
    -r                           Optional. Output Inference results as raw values.
    -t                           Optional. Probability threshold for person/vehicle/bike crossroad detections.
    -t_reid                      Optional. Cosine similarity threshold between two vectors for person reidentification.
    -no_show                     Optional. No show processed video.
    -auto_resize                 Optional. Enables resizable input with support of ROI crop & auto resize.
```

Running the application with an empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, to do inference on a GPU with the OpenVINO&trade; toolkit pre-trained models, run the following command:

```sh
./crossroad_camera_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/person-vehicle-bike-detection-crossroad-0078.xml -m_pa <path_to_model>/person-attributes-recognition-crossroad-0230.xml -m_reid <path_to_model>/person-reidentification-retail-0079.xml -d GPU
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.
In the default mode, the demo reports **Person Detection time** - inference time for the Person/Vehicle/Bike Detection network.

If Person Attributes Recognition or Person Reidentification Retail are enabled, the additional info below is reported also:
	* **Person Attributes Recognition time** - Inference time of Person Attributes Recognition averaged by the number of detected persons.
	* **Person Reidentification time** - Inference time of Person Reidentification averaged by the number of detected persons.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo has been tested on the following Model Downloader available topologies: 
>* `person-attributes-recognition-crossroad-0230`
>* `person-reidentification-retail-0079`
>* `person-vehicle-bike-detection-crossroad-0078`
> Other models may produce unexpected results on these devices.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
