# Crossroad Camera C++ Demo

![example](./crossroad_camera.gif)

This demo provides an inference pipeline for person detection, recognition and reidentification. The demo uses Person Detection network followed by the Person Attributes Recognition and Person Reidentification Retail networks applied on top of the detection results. You can use a set of the following pre-trained models with the demo:

* `person-vehicle-bike-detection-crossroad-0078`, which is a primary detection network for finding the persons (and other objects if needed)
* `person-attributes-recognition-crossroad-0230`, which is executed on top of the results from the first network and
reports person attributes like gender, has hat, has long-sleeved clothes
* `person-reidentification-retail-0277`, which is executed on top of the results from the first network and prints
a vector of features for each detected person. This vector is used to conclude if it is already detected person or not.

For more information about the pre-trained models, refer to the [model documentation](../../../models/intel/index.md).

Other demo objectives are:

* Images/Video/Camera as inputs, via OpenCV\*
* Example of simple networks pipelining: Person Attributes and Person Reidentification networks are executed on top of
the Person Detection results
* Visualization of Person Attributes and Person Reidentification (REID) information for each detected person

## How It Works

On startup, the application reads command line parameters and loads the specified networks. The Person Detection
network is required, and the other two are optional.

Upon getting a frame from the OpenCV VideoCapture, the application performs inference of Person Detection network, then performs another
two inferences of Person Attributes Recognition and Person Reidentification Retail networks if they were specified in the
command line, and displays the results.

If the Person Reidentification Retail network is specified, the resulting vector is generated for each detected person. This vector is
compared one-by-one with all previously detected persons vectors using cosine similarity algorithm. If comparison result
is greater than the specified (or default) threshold value, it is concluded that the person was already detected and a known
REID value is assigned. Otherwise, the vector is added to a global list, and new REID value is assigned.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with the `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Preparing to Run

For demo input image or video files you may refer to [Media Files Available for Demos](../../README.md#Media-Files-Available-for-Demos).
The list of models supported by the demo is in `<omz_dir>/demos/crossroad_camera_demo/cpp/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/downloader/README.md) and Converter to download and, if necessary, convert models to OpenVINO Inference Engine format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
python3 <omz_dir>/tools/downloader/downloader.py --list models.lst
```

An example of using the Model Converter:

```sh
python3 <omz_dir>/tools/downloader/converter.py --list models.lst
```

### Supported Models

* person-attributes-recognition-crossroad-0230
* person-attributes-recognition-crossroad-0234
* person-attributes-recognition-crossroad-0238
* person-reidentification-retail-0277
* person-reidentification-retail-0286
* person-reidentification-retail-0287
* person-reidentification-retail-0288
* person-vehicle-bike-detection-crossroad-0078
* person-vehicle-bike-detection-crossroad-1016

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
crossroad_camera_demo [OPTION]
Options:

    -h                           Print a usage message.
    -i                           Required. An input to process. The input must be a single image, a folder of images, video file or camera id.
    -loop                        Optional. Enable reading the input in a loop.
    -o "<path>"                  Optional. Name of output to save.
    -limit "<num>"               Optional. Number of frames to store in output. If 0 is set, all frames are stored.
    -m "<path>"                  Required. Path to the Person/Vehicle/Bike Detection Crossroad model (.xml) file.
    -m_pa "<path>"               Optional. Path to the Person Attributes Recognition Crossroad model (.xml) file.
    -m_reid "<path>"             Optional. Path to the Person Reidentification Retail model (.xml) file.
      -l "<absolute_path>"       Optional. For MKLDNN (CPU)-targeted custom layers, if any. Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"       Optional. For clDNN (GPU)-targeted custom kernels, if any. Absolute path to the xml file with the kernels desc.
    -d "<device>"                Optional. Specify the target device for Person/Vehicle/Bike Detection. The list of available devices is shown below. Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_pa "<device>"             Optional. Specify the target device for Person Attributes Recognition. The list of available devices is shown below. Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -d_reid "<device>"           Optional. Specify the target device for Person Reidentification Retail. The list of available devices is shown below. Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
    -pc                          Optional. Enables per-layer performance statistics.
    -r                           Optional. Output Inference results as raw values.
    -t                           Optional. Probability threshold for person/vehicle/bike crossroad detections.
    -t_reid                      Optional. Cosine similarity threshold between two vectors for person reidentification.
    -no_show                     Optional. Don't show output.
    -auto_resize                 Optional. Enables resizable input with support of ROI crop & auto resize.
    -u                           Optional. List of monitors to show initially.
    -person_label                Optional. The integer index of the objects' category corresponding to persons (as it is returned from the detection network, may vary from one network to another). The default value is 1.
```

Running the application with an empty list of options yields the usage message given above and an error message.

For example, to do inference on a GPU with the OpenVINO&trade; toolkit pre-trained models, run the following command:

```sh
./crossroad_camera_demo -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/person-vehicle-bike-detection-crossroad-0078.xml -m_pa <path_to_model>/person-attributes-recognition-crossroad-0230.xml -m_reid <path_to_model>/person-reidentification-retail-0079.xml -d GPU
```

> **NOTE**: The detection network returns as the result a set of detected objects, where each detected object consists of a bounding box and an index of the object's category (person/vehicle/bike). The demo runs Person Attributes Recognition and Person Reidentification networks only for the bounding boxes that has the category "person".
> Since different detection networks may have different category index corresponding to the category "person", this index may be pointed by the command line parameter `-person_label`.
> Please, note that
> * the model `person-vehicle-bike-detection-crossroad-0078` returns for persons the category index 1, it is the default value for the demo
> * the model `person-vehicle-bike-detection-crossroad-1016` returns for persons the category index 2, so for the demo to work correctly, the command line parameter `-person_label 2` should be added.

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.
In the default mode, the demo reports **Person Detection time** - inference time for the Person/Vehicle/Bike Detection network.

If Person Attributes Recognition or Person Reidentification Retail are enabled, the additional info below is reported also:

* **Person Attributes Recognition time** - Inference time of Person Attributes Recognition averaged by the number of detected persons.
* **Person Reidentification time** - Inference time of Person Reidentification averaged by the number of detected persons.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
