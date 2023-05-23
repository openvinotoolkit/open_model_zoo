# Handwritten Text Recognition Demo

This example demonstrates an approach to recognize handwritten Japanese, simplified Chinese, and English text lines using OpenVINO™. For Japanese, this demo supports all the characters in datasets [Kondate](http://web.tuat.ac.jp/~nakagawa/database/en/kondate_about.html) and [Nakayosi](http://web.tuat.ac.jp/~nakagawa/database/en/about_nakayosi.html). For simplified Chinese, it supports the characters in [SCUT-EPT](https://github.com/HCIILAB/SCUT-EPT_Dataset_Release). For English, it supports the characters in [GNHK](https://www.amazon.science/publications/gnhk-a-dataset-for-english-handwriting-in-the-wild).
## How It Works

The demo workflow is the following:

The demo first reads an image and performs the preprocessing such as resize and padding. Then after loading model to the plugin, the inference will start. After decoding the returned indexes into characters, the demo will display the predicted text.

## Preparing to Run

The list of models supported by the demo is in `<omz_dir>/demos/handwritten_text_recognition_demo/python/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO IR format (\*.xml + \*.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```

### Supported Models

* handwritten-japanese-recognition-0001
* handwritten-simplified-chinese-recognition-0001
* handwritten-english-recognition-0001

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the application with the `-h` option yields the following usage message:

```
usage: handwritten_text_recognition_demo.py [-h] -m MODEL -i INPUT [-d DEVICE]
                                            [-ni NUMBER_ITER] [-cl CHARLIST]
                                            [-dc DESIGNATED_CHARACTERS]
                                            [-tk TOP_K]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to an image to infer
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU or HETERO is acceptable. The
                        demo will look for a suitable plugin for device
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
  -ob OUTPUT_BLOB, --output_blob OUTPUT_BLOB
                        Optional. Name of the output layer of the model.
                        Default is None, in which case the demo will read
                        the output name from the model, assuming there is
                        only 1 output layer
```

The decoding char list files provided within Open Model Zoo and for Japanese it is the `<omz_dir>/data/dataset_classes/kondate_nakayosi.txt` file, while for Simplified Chinese it is the `<omz_dir>/data/dataset_classes/scut_ept.txt` file, and for English it is the `<omz_dir>/data/dataset_classes/gnhk.txt` file. For example, to do inference on a CPU with the OpenVINO&trade; toolkit pre-trained `handwritten-japanese-recognition-0001` model, run the following command:

```sh
python handwritten_text_recognition_demo.py \
  -d CPU \
  -i data/handwritten_japanese_test.png \
  -m <path_to_model>/handwritten-japanese-recognition-0001.xml
  -cl <omz_dir>/data/dataset_classes/kondate_nakayosi.txt \
```

When the `designated_characters` argument is provided, if the output character is not included in the designated characters, the script will check Top k steps in looking up the decoded character, until a designated one is found. By doing so, the output character will be restricted to a designated region. K is set to 20 by default.

For example, if you want to restrict the output characters to only digits and hyphens, you need to provide the path to the designated character file, for example `digit_hyphen.txt`. Then the script will perform a post-filtering processing on the output characters, but please note that it is possible that other characters are still allowed if none of designated characters are in the first K chosen elements. The mentioned characters text file located in the `data` subfolder of this demo.

The example command line for use pre-trained `handwritten-simplified-chinese-recognition-0001` model and `designated_charcters`option:

```sh
python handwritten_text_recognition_demo.py \
  -i data/handwritten_simplified_chinese_test.png \
  -m <path_to_model>/handwritten-simplified-chinese-recognition-0001.xml \
  -cl <omz_dir>/data/dataset_classes/scut_ept.txt \
  -dc data/digit_hyphen.txt
```

## Demo Output

The application uses the terminal to show resulting recognition text.
The demo reports

* **Latency**: total processing time required to process input data (from reading the data to displaying the results).

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
