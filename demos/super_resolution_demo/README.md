# Super Resolution Demo

This topic demonstrates how to run Super Resolution demo application, which
reconstructs the high resolution image from the original low resolution one.

The corresponding pre-trained model is delivered with the product:

* `single-image-super-resolution-0034`, which is the primary and only model that
  performs super resolution 4x upscale on a 200x200 image

For details on the model, please refer to the description in the
`deployment_tools/intel_models` folder of the OpenVINO&trade; toolkit
installation directory.

## How It Works

On the start-up, the application reads command-line parameters and loads the
specified network. After that, the application reads a 200x200 input image and
performs 4x upscale using super resolution.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./super_resolution_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

super_resolution_demo [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Path to an image.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -pp "<path>"            Path to a plugin folder.
    -d "<device>"           Specify the target device to infer on (CPU, GPU, FPGA, or MYRIAD). The demo will look for a suitable plugin for the specified device.
    -ni "<integer>"         Number of iterations (default value is 1)
    -pc                     Enable per-layer performance report

```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public models or a pre-trained and optimized model delivered with the package:

* `<INSTAL_DIR>/deployment_tools/intel_models/single-image-super-resolution-0034`

To do inference on CPU using a trained model, run the following command:

```sh
./super_resolution_demo -i <path_to_image>/image.bmp -m <path_to_model>/model.xml
```

**NOTE**: Public models should be first converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer](./docs/Model_Optimizer_Developer_Guide/Deep_Learning_Model_Optimizer_DevGuide.md) tool.

### Demo Output

The application outputs a reconstructed high-resolution image and saves it in
the current working directory as `*.bmp` file with `sr` prefix.

## See Also
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
