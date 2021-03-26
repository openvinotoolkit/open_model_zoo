# Handwritten Text Recognition Demo
This example demonstrates an approach to recognize handwritten Japanese and simplified Chinese text lines using OpenVINO™. For Japanese, this demo supports all the characters in datasets [Kondate](http://web.tuat.ac.jp/~nakagawa/database/en/kondate_about.html) and [Nakayosi](http://web.tuat.ac.jp/~nakagawa/database/en/about_nakayosi.html). For simplified Chinese, it supports the characters in [SCUT-EPT](https://github.com/HCIILAB/SCUT-EPT_Dataset_Release).

## How It Works

The demo expects the following model in the Intermediate Representation (IR) format:

   * handwritten-japanese-recognition-0001
   * handwritten-simplified-chinese-recognition-0001


It can be your own models or pre-trained model from OpenVINO Open Model Zoo.
In the `models.lst` are the list of appropriate models for this demo
that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).


The demo workflow is the following:

The demo first reads an image and performs the preprocessing such as resize and padding. Then after loading model to the plugin, the inference will start. After decoding the returned indexes into characters, the demo will display the predicted text.

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
usage: handwritten_text_recognition_demo.py [-h] -m MODEL -i INPUT [-d DEVICE]
                                            [-ni NUMBER_ITER] [-cl CHARLIST]
                                            [-dc DESIGNATED_CHARACTERS]
                                            [-tk TOP_K]

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
                        Path to the decoding char list file. Default is for
                        Japanese
  -dc DESIGNATED_CHARACTERS, --designated_characters DESIGNATED_CHARACTERS
                        Optional. Path to the designated character file
  -tk TOP_K, --top_k TOP_K
                        Optional. Top k steps in looking up the decoded
                        character, until a designated one is found
```


For example:
```
python handwritten_text_recognition_demo.py -i data/handwritten_japanese_test.png -m <path_to_model>/model.xml

```
When the `designated_characters` argument is provided, if the output character is not included in the designated characters, the script will check Top k steps in looking up the decoded character, until a designated one is found. By doing so, the output character will be restricted to a designated region. K is set to 20 by default.

For example, if we want to restrict the output characters to only digits and hyphens, we need to provide the path to the designated character file, e.g. `digit_hyphen.txt`. Then the script will perform a post-filtering processing on the output characters, but please note that it is possible that other characters are still allowed if none of `digit_hyphen.txt` is in first K chosen elements.

The command line:

```
python handwritten_text_recognition_demo.py -i data/handwritten_japanese_test.png -m <path_to_model>/model.xml -dc data/digit_hyphen.txt

```
## Demo Output
The application uses the terminal to show resulting recognition text and inference performance.


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
