# Crossroad Camera Demo

This demo provides an inference pipeline for persons' detection, recognition and reidentification. The demo uses Person Detection network followed by the Person Attributes Recognition and Person Reidentification Retail networks applied on top of the detection results. The corresponding pre-trained models are delivered with the product:

* `person-vehicle-bike-detection-crossroad-0078`, which is a primary detection network for finding the persons (and other objects if needed)
* `person-attributes-recognition-crossroad-0031`, which is executed on top of the results from the first network and
reports person attributes like gender, has hat, has long-sleeved clothes
* `person-reidentification-retail-0079`, which is executed on top of the results from the first network and prints
a vector of features for each detected person. This vector is used to conclude if it is already detected person or not.

For details on the models, please refer to the descriptions in the `deployment_tools/intel_models` folder of the
 OpenVINO&trade; toolkit installation directory.

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
      -l "<absolute_path>"       Optional. For MKLDNN (CPU)-targeted custom layers, if any. Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"       Optional. For clDNN (GPU)-targeted custom kernels, if any. Absolute path to the xml file with the kernels desc.
    -d "<device>"                Optional. Specify the target device for Person/Vehicle/Bike Detection (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -d_pa "<device>"             Optional. Specify the target device for Person Attributes Recognition (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -d_reid "<device>"           Optional. Specify the target device for Person Reidentification Retail (CPU, GPU, FPGA, MYRIAD, or HETERO).
    -pc                          Optional. Enables per-layer performance statistics.
    -r                           Optional. Output Inference results as raw values.
    -t                           Optional. Probability threshold for person/vehicle/bike crossroad detections.
    -t_reid                      Optional. Cosine similarity threshold between two vectors for person reidentification.
    -no_show                     Optional. No show processed video.
    -auto_resize                 Optional. Enables resizable input with support of ROI crop & auto resize.
```

Running the application with an empty list of options yields the usage message given above and an error message.

To run the demo, you can use public models or a set of pre-trained and optimized models delivered with the package:

* `<INSTAL_DIR>/deployment_tools/intel_models/person-vehicle-bike-detection-crossroad-0078`
* `<INSTAL_DIR>/deployment_tools/intel_models/person-attributes-recognition-crossroad-0031`
* `<INSTAL_DIR>/deployment_tools/intel_models/person-reidentification-retail-0079`

For example, to do inference on a GPU with the OpenVINO&trade; toolkit pre-trained models, run the following command:

```sh
./crossroad_camera_demo -i <path_to_video>/inputVideo.mp4 -m person-vehicle-bike-detection-crossroad-0078.xml -m_pa person-attributes-recognition-crossroad-0031.xml -m_reid person-reidentification-retail-0079.xml -d GPU
```
**NOTE**: Public models should be first converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer](./docs/Model_Optimizer_Developer_Guide/Deep_Learning_Model_Optimizer_DevGuide.md) tool.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.
In the default mode, the demo reports **Person Detection time** - inference time for the Person/Vehicle/Bike Detection network.

If Person Attributes Recognition or Person Reidentification Retail are enabled, the additional info below is reported also:
	* **Person Attributes Recognition time** - Inference time of Person Attributes Recognition averaged by the number of detected persons.
	* **Person Reidentification time** - Inference time of Person Reidentification averaged by the number of detected persons.


## See Also
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)