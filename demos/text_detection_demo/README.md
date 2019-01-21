# Text Detection Demo

The demo shows an example of using a single neural network to detect printed text rotated at any angle in various environment. The corresponding pre-trained model is delivered with the product:

* `text-detection-0001`, which is a detection network for finding text.

### How It Works

On the start-up, the application reads command line parameters and loads one network to the Inference Engine for execution. Upon getting an image, it performs inference of text detection and prints the result as four points (`x1`, `y1`), (`x2`, `y2`), (`x3`, `y3`), (`x4`, `y4`) for each text bounding box.

### Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./text_detection_demo -h

text_detection_demo [OPTION]
Options:

    -h                           Print a usage message.
    -i "<path>"                  Required. Path to an image file.
    -m "<path>"                  Required. Path to the Text Detection model (.xml) file.
    -d "<device>"                Optional. Specify the target device to infer on: CPU, GPU, FPGA, or MYRIAD. The demo will look for a suitable plugin for a specified device.
    -l "<absolute_path>"         Optional. Absolute path to a shared library with the CPU kernels implementation for custom layers.
    -c "<absolute_path>"         Optional. Absolute path to the GPU kernels implementation for custom layers.
    -no_show                     Optional. If it is true, then detected text will not be shown on image frame. By default, it is false.
    -r                           Optional. Output Inference results as raw values.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use the following pre-trained and optimized model:

* `open_model_zoo/intel_models/text-detection-0001`

For example, use the following command line command to run the application:
```sh
./text_detection_demo -m <path_to_model> -i <path_to_image>
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes.

## See Also
* [Using Inference Engine Demo](../Readme.md)
