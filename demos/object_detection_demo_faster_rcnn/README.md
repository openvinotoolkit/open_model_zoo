# Object Detection Faster R-CNN C++ Demo

This topic demonstrates how to run the Object Detection demo application, which does inference using object detection
networks like Faster R-CNN on Intel® Processors and Intel® HD Graphics.

## Downloading and Converting Caffe* Model
VGG16-Faster-RCNN is a public CNN that can be easily obtained from GitHub:

1. Download <code>test.prototxt</code> from [https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt](https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt)
2. Download pre-trained models from [https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0](https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0)
3. Unzip the archive. You will need the <code>VGG16_faster_rcnn_final.caffemodel</code> file.

To convert the source model, run the Model Optimizer.
You can use the following command to convert the source model:
```sh
python3 ${MO_ROOT_PATH}/mo_caffe.py --input_model <path_to_model>/VGG16_faster_rcnn_final.caffemodel --input_proto <path_to_model>/deploy.prototxt
```

For documentation on how to convert Caffe models, refer to [Converting a Caffe Model](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html).

## How It Works

Upon the start-up, the demo application reads command line parameters and loads a network and an image to the Inference
Engine plugin. When inference is done, the application creates an
output image and outputs data to the standard output stream.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./object_detection_demo_faster_rcnn -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

object_detection_demo_faster_rcnn [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path>"               Required. Path to a .bmp image.
    -m "<path>"               Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for CPU custom layers. Absolute path to a shared library with the kernel implementations.
      -c "<absolute_path>"    Required for GPU custom kernels. Absolute path to the .xml file with the kernel descriptions.
    -d "<device>"             Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The demo will look for a suitable plugin for a specified device.
    -bbox_name "<string>"     Optional. The name of output box prediction layer. Default value is "bbox_pred"
    -proposal_name "<string>" Optional. The name of output proposal layer. Default value is "proposal"
    -prob_name "<string>"     Optional. The name of output probability layer. Default value is "cls_prob"
    -p_msg                    Optional. Enables messages from a plugin
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

You can use the following command to do inference on CPU on an image using a trained Faster R-CNN network:
```sh
./object_detection_demo_faster_rcnn -i <path_to_image>/inputImage.bmp -m <path_to_model>/faster-rcnn.xml -d CPU
```

## Demo Output

The application outputs an image (`out_0.bmp`) with detected objects enclosed in rectangles. It outputs the list of classes
of the detected objects along with the respective confidence values and the coordinates of the
rectangles to the standard output stream.

> **NOTE**: On VPU devices (Intel® Movidius™ Neural Compute Stick, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs) this demo is not supported with any of the Model Downloader available topologies. Other models may work incorrectly on these devices as well.

## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Downloader](../../tools/downloader/README.md)
* [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)
* [Converting a Caffe Model](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html)
