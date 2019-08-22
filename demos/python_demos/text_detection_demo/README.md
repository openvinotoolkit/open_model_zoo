# Text Detection Python3 demo

Decoder for Intel's version of PixelLink -- [text-detection-0001](https://github.com/opencv/open_model_zoo/blob/2019/intel_models/text-detection-0001/description/text-detection-0001.md).

You will need an OpenCV compiled with Inference Engine to use this, e.g. [opencv-python-inference-engine==4.1.0.3](https://pypi.org/project/opencv-python-inference-engine/4.1.0.3/).

## How it works

Upon the start-up the demo application reads command line parameters and loads a network and an image to the Inference Engine plugin. When inference is done, the application creates an output image.

## Running

Running the `python3 text_detection_demo.py -h` yields the following usage message:

```
usage: text_detection_demo.py [-h] -i IMAGE_PATH -m MODEL_PATH

optional arguments:
  -h, --help     show this help message and exit
  -i IMAGE_PATH  path to input image
  -m MODEL_PATH  path to model's XML file
```

**Example of in code usage:**

```python
import cv2
from text_detection_demo import PixelLinkDecoder


dcd = PixelLinkDecoder()
td = cv2.dnn.readNet('./text-detection-0001.xml','./text-detection-0001.bin')

img = cv2.imread('tmp.jpg')
blob = cv2.dnn.blobFromImage(img, 1, (1280,768))

td.setInput(blob)
a, b = td.forward(td.getUnconnectedOutLayersNames())

dcd.load(img, a, b)
dcd.decode()  # results are in dcd.bboxes
dcd.plot_result_pyplot(img)
```

## Demo output

Image with plotted bounding boxes around a text fields.

## See Also

* [Using Open Model Zoo demos](https://github.com/opencv/open_model_zoo/tree/2019/demos/README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/2019_R1/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2019/model_downloader)
* [PixelLink original paper](https://arxiv.org/pdf/1801.01315.pdf)
* [PixelLink original code](https://github.com/ZJULearning/pixel_link)
* [PixelLink in Keras](https://github.com/opconty/pixellink_keras)
