# TensorFlow* Object Detection Mask R-CNNs Segmentation Demo

This topic demonstrates how to run the Segmentation demo application, which does inference using image segmentation networks created with Object Detection API. Note that batch size 1 is supported only.

The demo has a post-processing part that gathers masks arrays corresponding to bounding boxes with high probability taken from the Detection Output layer. Then the demo produces picture with identified masks.

## Running

Running the application with the <i>-h</i> option yields the following usage message:
```sh
./mask_rcnn_demo -h
InferenceEngine: 
    API version ............ <version>
    Build .................. <number>

mask_rcnn_demo [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Path to an .bmp image.
    -m "<path>"             Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"    Required for clDNN (GPU)-targeted custom kernels.Absolute path to the xml file with the kernels desc.
    -pp "<path>"            Path to a plugin folder.
    -d "<device>"           Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. The demo will look for a suitable plugin for a specified device (CPU by default)
    -ni "<integer>"         Number of iterations (default 1)
    -detection_output_name "<string>" The name of detection output layer (default: detection_output)
    -masks_name "<string>" The name of masks layer (default: masks)
    -pc                     Enables per-layer performance report
```

Running the application with the empty list of options yields the usage message given above and an error message.

You can use the following command to do inference on Intel&reg; Processors on an image using a trained network:
```sh
./mask_rcnn_demo -i <path_to_image>/inputImage.bmp -m <path_to_model>/faster_rcnn.xml
```

### Outputs

The application output is a segmented image (out.png).

### How it works

Upon the start-up the demo application reads command line parameters and loads a network and an image to the Inference Engine plugin. When inference is done, the application creates an output image.

## See Also 
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
