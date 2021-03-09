# Face Detection MTCNN Python* Demo

This demo demonstrates how to run `mtcnn` model using OpenVINO&trade;.

For more information about the pre-trained models, refer to the [model documentation](../../../models/public/index.md).

## How It Works

The demo application expects `mtcnn` models in the Intermediate Representation (IR) format.

The demo workflow is the following:

1. Use mtcnn model to detect face position and feature points on input image and display detection results in application window.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Run the application with the `-h` option to see available command line parameters:

```
python3 face_detection_mtcnn_demo.py -h
```

The command yields the following usage message:

```
usage: face_detection_mtcnn_demo.py [-h] -i INPUT -m_p "<path>" -m_r "<path>"
                                    -m_o "<path>" [-th "<num>"]
                                    [-d "<device>"] [--loop] [--no_show]
                                    [-o OUTPUT] [-limit OUTPUT_LIMIT]
                                    [-u UTILIZATION_MONITORS]

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. Path to a test image file.
  -m_p "<path>", --model_pnet "<path>"
                        Required. Path to an .xml file with a pnet model.
  -m_r "<path>", --model_rnet "<path>"
                        Required. Path to an .xml file with a rnet model.
  -m_o "<path>", --model_onet "<path>"
                        Required. Path to an .xml file with a onet model.
  -th "<num>", --threshold "<num>"
                        Optional. The threshold to define the face is
                        recognized or not.
  -d "<device>", --device "<device>"
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU
  --loop                Optional. Enable reading the input in a loop.
  --no_show             Optional. Don't show output
  -o OUTPUT, --output OUTPUT
                        Optional. Name of output to save.
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output. If 0 is
                        set, all frames are stored.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.
```

Running the application with an empty list of options yields the short version of the usage message and an error message.

To run the demo, you can use public pre-trained MTCNN models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in [models.lst](./models.lst).

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
