# Face Detection MTCNN Python* Demo

This demo demonstrates how to run `mtcnn` model using OpenVINO&trade;.

For more information about the pre-trained models, refer to the [model documentation](../../../models/public/index.md).

## How It Works

The demo application expects `mtcnn` models in the Intermediate Representation (IR) format.

The demo workflow is the following:

1. Use mtcnn model to detect face position and feature points on input image and display detection results in application window.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Run the application with the `-h` option to see the following usage message:

```
usage: face_detection_mtcnn_demo.py \
    [-h] \
    [-i INPUT] \
    [-m_p MODEL_PNET] \
    [-m_r MODEL_RNET] \
    [-m_o MODEL_ONET] \
    [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. Path to a test image file.
  -m_p MODEL_PNET, --model_pnet MODEL_PNET
                        Required. Path to an .xml file with a pnet model.
  -m_r MODEL_RNET, --model_rnet MODEL_RNET
                        Required. Path to an .xml file with a rnet model.
  -m_o MODEL_ONET, --model_onet MODEL_ONET
                        Required. Path to an .xml file with a onet model.
  -th THRESHOLD, --threshold THRESHOLD
                        Optional. The threshold to define the face is
                        recognized or not.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU

```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in [models.lst](./models.lst).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
To run the demo, please provide paths to the model in the IR format, and to an input video or image(s):
```bash
python face_detection_mtcnn_demo.py \
    -i /home/user/image_name.jpg \
    -m_p <path_to_model>/mtcnn-p.xml \
    -m_o <path_to_model>/mtcnn-o.xml \
    -m_r <path_to_model>/mtcnn-r.xml \
    -th 0.7
```

## Demo Output

The application uses OpenCV to display found faces' boundary, feature points and current inference performance.

![](./test.jpg)

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
