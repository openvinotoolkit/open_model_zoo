# Handwritten Japanese Recognition Demo
This example demonstrates an approach to recognize handwritten japanese text lines using OpenVINO™. This model supports all the characters in datasets [Kondate](http://web.tuat.ac.jp/~nakagawa/database/en/kondate_about.html) and [Nakayosi](http://web.tuat.ac.jp/~nakagawa/database/en/about_nakayosi.html).

## How It Works

The demo expects the following model in the Intermediate Representation (IR) format:

   * handwritten-japanese-recognition-0001

It can be your own models or pre-trained model from OpenVINO Open Model Zoo.
In the `models.lst` are the list of appropriate models for this demo
that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).


The demo workflow is the following:

The demo first reads an image and performs the preprocessing such as resize and padding. Then after loading model to the plugin, the inference will start. After decoding the returned indexes into chars, the demo will display the predicted text.

### Installation and dependencies

The demo depends on:
- opencv-python
- numpy

To install all the required Python modules you can use:

``` sh
pip install -r requirements.txt
```

### Command line arguments
```
usage: handwritten_japanese_recognition_demo.py [-h] -m MODEL -i INPUT
                                                [-d DEVICE] [-ni NUMBER_ITER]
                                                [-cl CHARLIST]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to an image to infer
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU
  -ni NUMBER_ITER, --number_iter NUMBER_ITER
                        Optional. Number of inference iterations
  -cl CHARLIST, --charlist CHARLIST
                        Path to the decoding char list file
```


For example:
```
python handwritten_japanese_recognition_demo.py -i data/test.png -m path/to/ir_xml/model.xml

```
## Demo Output
The application uses the terminal to show resulting recognition text and inference performance.


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
