# Python* Model API package

Model API package is a set of wrapper classes for particular tasks and model architectures, simplifying data preprocess and postprocess as well as routine procedures (model loading, asynchronous execution, etc...)
Model API wrappers hide the specific code inside and work as black-box: application feeds model class with input data, then the model returns post-processed output data in user-friendly format.

## Package structure

The Python* Model API consists of 3 libraries:
* _adapters_ - implement a common interface to allow Model API wrappers usage with different executors: OpenVINO, ONNX, etc. See [Model API adapters](#model-api-adapters) section
* _models_ - implement wrappers for each architecture. See [Model API Wrappers](#model-api-wrappers) section
* _pipelines_ - implement pipelines for model inference and manage the synchronous/asynchronous execution. See [Model API Pipelines](#model-api-pipelines) section

## Building Python* Model API package
For installation Python (version 3.6 or higher) is required. The installation is available from source.

Use the following command to install Python* Model API from source:
```sh
pip install <omz_dir>/demos/common/python
```

Alternatively, you can generate the package using a wheel. Follow the steps below:
1. Build the wheel.

```sh
python <omz_dir>/demos/common/python/setup.py bdist_wheel
```
The wheel should appear in the dist folder.
Name example: `openmodelzoo_modelapi-0.0.0-py3-none-any.whl`

2. Install the package in the clean environment with `--force-reinstall` key.
```sh
pip install openmodelzoo_modelapi-0.0.0-py3-none-any.whl --force-reinstall
```

To verify the package is installed, you might use the following command:
```sh
python -c "from openvino.model_zoo import model_api"
```

> **NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`. You may also need to [install pip](https://pip.pypa.io/en/stable/installation/).
> For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-pip`.

## Model API Wrappers

The Python* Model API package suggests ready-to-use model wrappers, which implement standardized preprocessing/postprocessing functions per "task type" and might be reused in applications as "black-box" models.
Also, the simple wrapper interface allows the creation of custom wrappers covering different architectures.

The following tasks can be solved with wrappers usage:

| Task type                  | Model API wrappers |
|----------------------------|--------------------|
| Background Matting         | <ul><li>`VideoBackgroundMatting`</li><li>`ImageMattingWithBackground`</li></ul> |
| Classification             | <ul><li>`Classification`</li></ul> |
| Deblurring                 | <ul><li>`Deblurring`</li></ul> |
| Human Pose Estimation      | <ul><li>`HpeAssociativeEmbedding`</li><li>`OpenPose`</li></ul> |
| Instance Segmentation      | <ul><li>`MaskRCNNModel`</li><li>`YolactModel`</li></ul> |
| Monocular Depth Estimation | <ul><li> `MonoDepthModel`</li></ul> |
| Named Entity Recognition   | <ul><li>`BertNamedEntityRecognition`</li></ul> |
|  Object Detection          | <ul><li>`CenterNet`</li><li>`DETR`</li><li>`CTPN`</li><li>`FaceBoxes`</li><li>`RetinaFace`</li><li>`RetinaFacePyTorch`</li><li>`SSD`</li><li>`UltraLightweightFaceDetection`</li><li>`YOLO`</li><li>`YoloV3ONNX`</li><li>`YoloV4`</li><li>`YOLOF`</li><li>`YOLOX`</li></ul> |
| Question Answering         |  <ul><li>`BertQuestionAnswering`</li></ul> |
| Salient Object Detection   |  <ul><li>`SalientObjectDetectionModel`</li></ul> |
| Semantic Segmentation      |  <ul><li>`SegmentationModel`</li></ul> |

## Model API Adapters

Model API wrappers are executor-agnostic, meaning it does not implement the specific model inference or model loading, instead it can be used with different executors having the implementation of common interface methods in adapter class respectively.

Currently, `OpenvinoAdapter` and `OVMSAdapter` are supported.

#### OpenVINO executor

`OpenvinoAdapter` hides the OpenVINO™ toolkit API, which allows Model API wrappers launching with models represented in Intermediate Representation (IR) format.
It accepts a path to either `xml` model file or `onnx` model file.

For OpenVINO executor employment, you need to install the requirements:
```sh
pip install <omz_dir>/demos/common/python/requirements_openvino.txt
```

#### OpenVINO Model Server executor

`OVMSAdapter` hides the OpenVINO Model Server python client API, which allows Model API wrappers launching with models served by OVMS.

Refer to __[`OVMSAdapter`](adapters/ovms_adapter.md)__ to learn about running demos with OVMS.

For OpenVINO Model Server executor employment, you need to install the requirements:
```sh
pip install <omz_dir>/demos/common/python/requirements_ovms.txt
```

## Model API Pipelines

Model API Pipelines represent the high-level wrappers upon the input data and accessing model results management.
It performs the data submission for model inference, verification of inference status, whether the result is ready or not, and results accessing.

The `AsyncPipeline` is available, which handles the asynchronous execution of a single model.

## Ready-to-use Model API solutions

To apply Model API wrappers in custom applications, learn the provided example of common scenario of how to use Python* Model API.

 In the example, the SSD architecture is used to predict bounding boxes on input image `"sample.png"`. The model execution is produced by `OpenvinoAdapter`, therefore we submit the path to the model's `xml` file. The model is loaded on a CPU device inside the adapter.

Once the SSD model wrapper instance is created, we get the predictions by the model in one line: `ssd_model(input_data)` - the wrapper performs the preprocess method, synchronous inference on OpenVINO side and postprocess method.

```python
import cv2
from openvino.model_zoo.model_api.models import SSD
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core


# a helper function for bboxes visualization
def draw_detections(image, detections):
    for detection in detections:
        class_id = int(detection.id)
        color = (255, 0, 0)
        det_label = '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return image


def main():
    input_data = cv2.imread("sample.png")
    model_path = "public/mobilenet-ssd/FP32/mobilenet-ssd.xml"

    model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")
    ssd_model = SSD(model_adapter, preload=True)

    results = ssd_model(input_data)

    image_with_bboxes = draw_detections(input_data, results)
    cv2.imshow('Detection Results', image_with_bboxes)
    key = cv2.waitKey(0)
    if key in {ord('q'), ord('Q'), 27}:
        return


if __name__ == '__main__':
    main()
```

To study the complex scenarios, refer to [Open Model Zoo Python* demos](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos), where the asynchronous inference is applied.
