# Text Detection C++ Demo

The demo shows an example of using neural networks to detect and recognize printed text rotated at any angle in various environment. You can use the following pre-trained models with the demo:

* `text-detection-0002`, which is a detection network for finding text.
* `text-recognition-0012`, which is a recognition network for recognizing text.
* `handwritten-score-recognition-0001`, which is a recognition network for recognizing handwritten score marks like `<digit>` or `<digit>.<digit>`.

For more information about the pre-trained models, refer to the [Open Model Zoo](https://github.com/opencv/open_model_zoo/tree/master/intel_models/index.md) repository on GitHub*.

## How It Works

On the start-up, the application reads command line parameters and loads one network to the Inference Engine for execution. Upon getting an image, it performs inference of text detection and prints the result as four points (`x1`, `y1`), (`x2`, `y2`), (`x3`, `y3`), (`x4`, `y4`) for each text bounding box.

If text recognition model is provided, the demo prints recognized text as well.  

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Specify Input Shapes** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./text_detection_demo -h

text_detection_demo [OPTION]
Options:

    -h                           Print a usage message.
    -i "<path>"                  Required. Path to an image or video file, to a text file with paths to images, or to a webcamera device node (for example, /dev/video0).
    -m_td "<path>"               Required. Path to the Text Detection model (.xml) file.
    -m_tr "<path>"               Required. Path to the Text Recognition model (.xml) file.
    -m_tr_ss "<value>"           Optional. Symbols set for Text Recognition model.
    -cc                          Optional. If it is true, then in case of absence of Text Detector, Text Reconition model will take a central image crop as an input, but not full frame. By default, it is false.
    -w_td "<value>"              Optional. Input image width for Text Detection model.
    -h_td "<value>"              Optional. Input image height for Text Detection model.
    -thr "<value>"               Optional. Specify a recognition confidence threshold. Text detection candidates with text recognition confidence below specified threshold are rejected.
    -cls_pixel_thr "<value>"     Optional. Specify a confidence threshold for pixel classification. Pixels with classification confidence below specified threshold are rejected.
    -link_pixel_thr "<value>"    Optional. Specify a confidence threshold for pixel linkage. Pixels with linkage confidence below specified threshold are not linked.
    -max_rect_num "<value>"      Optional. Maximum number of rectangles to recognize. If it is negative, number of rectangles to recognize is not limited.
    -dt "<input_data_type>"      Optional. Input data type: "image" (for a single image), "list" (for a text file where images paths are listed), "video" (for a saved video), "webcam" (for a webcamera device). By default, it is "image".
    -d_td "<device>"             Optional. Specify the target device for the Text Detection model to infer on: CPU, GPU. The demo will look for a suitable plugin for a specified device. By default, it is CPU.
    -d_tr "<device>"             Optional. Specify the target device for the Text Recognition model to infer on: CPU, GPU. The demo will look for a suitable plugin for a specified device. By default, it is CPU.
    -l "<absolute_path>"         Optional. Absolute path to a shared library with the CPU kernels implementation for custom layers.
    -c "<absolute_path>"         Optional. Absolute path to the GPU kernels implementation for custom layers.
    -no_show                     Optional. If it is true, then detected text will not be shown on image frame. By default, it is false.
    -r                           Optional. Output Inference results as raw values.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

For example, use the following command line command to run the application:
```sh
./text_detection_demo -m_td <path_to_text_detection_model> \
                      -m_tr <path_to_text_recognition_model> \
                      -i <path_to_image>
```

## Demo Output

The demo uses OpenCV to display the resulting frame with detections rendered as bounding boxes and text.

## See Also
* [Using Open Model Zoo demos](https://github.com/opencv/open_model_zoo/tree/master/demos/README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader)
