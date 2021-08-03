# Handwritten Chinese Table OCR Demo
## Description
This demo demonstrates using OpenCV and OpenVINO toolkit to recognize handwritten Chinese/Japanese tables and generate docx files with a GUI.


## How It Works
The demo uses the following model in the Intermediate Representation (IR) format:
   * handwritten-japanese-recognition-0001
   * handwritten-simplified-chinese-recognition-0001


This demo workflow mainly has two parts:

- The first part is the detection of the table. This part uses OpenCV to perform perspective transformation, and then detects the cells of the table.

- The second part uses the cell images detected in the first part for handwritten text recognition with OpenVINO. It first reads a cell image and performs the preprocessing such as further binarization, resize and padding, and then the inference will start. After decoding the returned indexes into characters, the demo will display the predicted text. Optionally, you can generate the predicted text into a docx file.

## Installation and dependencies

### OpenVINO Requirement

You should install OpenVINO Toolkit (version >= 2020.3).

For more OpenVINO toolkit installation information, please refer to [here](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html).

### Other Dependencies
This project requires Python 3.6.

The required Python modules can be installed using pip as follows:

```
pip install -r requirements.txt
```


## How to Run



Create models directory:
```
mkdir models
```

To download the models, please use [Model Downloader](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/downloader) to download, which is a component of OpenVINO toolkit.
After downloading is finished, put model files into `models`:

E.g.:

```
models/handwritten-simplified-chinese-recognition-0001.xml
models/handwritten-simplified-chinese-recognition-0001.bin
```

### Usage
```
python table_demo_main.py -h
```

```
usage: table_demo_main.py [-h] -m MODEL [-c CHARLIST] [-d DEVICE]
                          [-dc DESIGNATED_CHARACTERS] [-tk TOP_K]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -c CHARLIST, --charlist CHARLIST
                        Optional. Path to the decoding char list file. Default
                        is data/scut_ept_char_list.txt
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU
  -dc DESIGNATED_CHARACTERS, --designated_characters DESIGNATED_CHARACTERS
                        Optional. Path to the designated character file
  -tk TOP_K, --top_k TOP_K
                        Optional. Top k steps in looking up the decoded
                        character, until a designated one is found
```
To run the demo:
```
python table_demo_main.py -m models/handwritten-simplified-chinese-recognition-0001.xml
```

You will see that a GUI is started, open the handwritten text table you need to detect, and perform the detection and recognition in order. If there is a recognition error, you can manually modify it on the right side. Finally it can be saved as a docx file.

## Example Diagram
![demo](data/handwritten_table_recoginition_demo.gif)

## See Also:
[OpenVINO](https://docs.openvinotoolkit.org/)