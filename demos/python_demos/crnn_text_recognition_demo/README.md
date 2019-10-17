# CRNN Text Recognition Python* Demo 

This demo demonstrates how to make English text recognition based on CRNN (Convolutional Reccurent Neural Network) model. 

## How It Works

The demo application expects a CRNN model converted to the Intermediate Representation (IR) format using this instruction: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_CRNN_From_Tensorflow.html

As input the demo application takes a path to the singe image file.

The demo workflow is the following:

1. The demo application reads the input image.
2. Runs the CRNN model inference
3. Decodes the result using CTC beam search decoder
4. Matches the decoded lines with the given English alphabet. 
5. Shows the input picture and puts the detected text as it's title and prints it to the console. 

## Running

Run the application with the `-h` option to see the following usage message:

```

usage: crnn_text_recognition_demo.py [-h] -i "<path>" -m "<path>"
                                     [-d "<device>"] [-l "<absolute_path>"]

Options:
  -h, --help            Show this help message and exit.
  -i "<path>"           Required. Path to the input image.
  -m "<path>", --model "<path>"
                        Required. Path to an .xml file with a trained model.
  -d "<device>", --device "<device>"
                        Optional. Specify the target device to infer on: CPU,
                        GPU, FPGA, HDDL or MYRIAD. The demo will look for a
                        suitable plugin for device specified (by default, it
                        is CPU).
  -l "<absolute_path>", --cpu_extension "<absolute_path>"
                        Required for CPU custom layers. Absolute path to a
                        shared library with the kernels implementation.

```

Running the application with the empty list of options yields the usage message given above and an error message.

## Demo Output

The application uses OpenCV to display the input image, show the detected text as image title and prints it to the console.

## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)