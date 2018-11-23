# Object Detection Faster R-CNN Demo

This topic demonstrates how to run the Object Detection demo application, which does inference using object detection
networks like Faster R-CNN on Intel® Processors and Intel® HD Graphics.

## Downloading and Converting Caffe* Model
VGG16-Faster-RCNN is a public CNN that can be easily obtained from GitHub:

1. Download <code>test.prototxt</code> from <code>https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt</code>
2. Download pre-trained models from <code>https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0</code>
3. Unzip the archive. You will need <code>VGG16_faster_rcnn_final.caffemodel</code> file.

For correct converting the source model you should run the Model Optimizer. 
You can use the following command to convert the source model:
```sh
python3 ${MO_ROOT_PATH}/mo_caffe.py --input_model <path_to_model>/VGG16_faster_rcnn_final.caffemodel --input_proto <path_to_model>/deploy.prototxt
```

For documentation on how to convert Caffe models, refer to [Converting a Caffe Model](./docs/Model_Optimizer_Developer_Guide/prepare_trained_model/convert_model/Convert_Model_From_Caffe.md).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./object_detection_demo -h
InferenceEngine: 
    API version ............ <version>
    Build .................. <number>

object_detection_demo [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path>"               Required. Path to an .bmp image.
    -m "<path>"               Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.
      -c "<absolute_path>"    Required for clDNN (GPU)-targeted custom kernels. Absolute path to the xml file with the kernels desc.
    -pp "<path>"              Path to a plugin folder.
    -d "<device>"             Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. The demo will look for a suitable plugin for a specified device.
    -pc                       Enables per-layer performance report
    -ni "<integer>"           Number of iterations (default 1)
    -bbox_name "<string>"     The name of output box prediction layer (default: bbox_pred)
    -proposal_name "<string>" The name of output proposal layer (default: proposal)
    -prob_name "<string>"     The name of output probability layer (default: cls_prob)
    -p_msg                    Enables messages from a plugin
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can use the following command to do inference on Intel&reg; Processors on an image using a trained Faster R-CNN network:
```sh
./object_detection_demo -i <path_to_image>/inputImage.bmp -m <path_to_model>/faster-rcnn.xml -d CPU
```

### Outputs

The application outputs an image (<code>out_0.bmp</code>) with detected objects enclosed in rectangles. It outputs the list of classes 
of the detected objects along with the respective confidence values and the coordinates of the 
rectangles to the standard output stream.

### How it works

Upon the start-up the demo application reads command line parameters and loads a network and an image to the Inference
Engine plugin. When inference is done, the application creates an 
output image and outputs data to the standard output stream.

## Using This Demo with Intel's Person Detection model

This model has a non-default (for Faster-RCNN) output layer name. In order to score it correctly, you should add an option 
`--bbox_name detector/bbox/ave_pred` to the command line.

Usage example: 

```sh
./object_detection_demo -i <path_to_image>/people.jpg -m <INSTALL_DIR>/deployment_tools/intel_models/person-detection-retail-0001/FP32/person-detection-retail-0001.xml --bbox_name detector/bbox/ave_pred -d CPU
```

## See Also 
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
* [Converting a Model Using General Conversion Parameters](@ref ConvertGeneral)
* [Converting a Caffe Model](./docs/Model_Optimizer_Developer_Guide/prepare_trained_model/convert_model/Convert_Model_From_Caffe.md)
