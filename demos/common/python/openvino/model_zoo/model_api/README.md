# Python* Model API package

Model API package is a set of wrapper classes for particular tasks and model architectures, simplifying data preprocess and postprocess as well as routine procedures (model loading, asynchronous execution, etc...)
An application feeds model class with input data, then the model returns postprocessed output data in user-friendly format.

## Package structure

The Model API consists of 3 libraries:
* _adapters_ implements a common interface to allow Model API wrappers usage with different executors. See [Model API adapters](#model-api-adapters) section
* _models_ implements wrappers for Open Model Zoo models. See [Model API Wrappers](#model-api-wrappers) section
* _pipelines_ implements pipelines for model inference and manage the synchronous/asynchronous execution. See [Model API Pipelines](#model-api-pipelines) section

### Prerequisites

The package requires
- one of OpenVINO supported Python version (see OpenVINO documentation for the details)
- OpenVINO™ toolkit

If you build Model API package from source, you should install the OpenVINO™ toolkit. See the options:

Use installation package for [Intel® Distribution of OpenVINO™ toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit-download.html) or build the open-source version available in the [OpenVINO GitHub repository](https://github.com/openvinotoolkit/openvino) using the [build instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

Also, you can install the OpenVINO Python\* package via the command:
 ```sh
pip install openvino
 ```

## Installing Python* Model API package

Use the following command to install Model API from source:
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

## Model API Wrappers

The Model API package provides model wrappers, which implement standardized preprocessing/postprocessing functions per "task type" and incapsulate model-specific logic for usage of different models in a unified manner inside the application.

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
| Object Detection           | <ul><li>`CenterNet`</li><li>`DETR`</li><li>`CTPN`</li><li>`FaceBoxes`</li><li>`RetinaFace`</li><li>`RetinaFacePyTorch`</li><li>`SSD`</li><li>`UltraLightweightFaceDetection`</li><li>`YOLO`</li><li>`YoloV3ONNX`</li><li>`YoloV4`</li><li>`YOLOF`</li><li>`YOLOX`</li></ul> |
| Question Answering         |  <ul><li>`BertQuestionAnswering`</li></ul> |
| Salient Object Detection   |  <ul><li>`SalientObjectDetectionModel`</li></ul> |
| Semantic Segmentation      |  <ul><li>`SegmentationModel`</li></ul> |

## Model API Adapters

Model API wrappers are executor-agnostic, meaning it does not implement the specific model inference or model loading, instead it can be used with different executors having the implementation of common interface methods in adapter class respectively.

Currently, `OpenvinoAdapter` and `OVMSAdapter` are supported.

#### OpenVINO Adapter

`OpenvinoAdapter` hides the OpenVINO™ toolkit API, which allows Model API wrappers launching with models represented in Intermediate Representation (IR) format.
It accepts a path to either `xml` model file or `onnx` model file.

#### OpenVINO Model Server Adapter

`OVMSAdapter` hides the OpenVINO Model Server python client API, which allows Model API wrappers launching with models served by OVMS.

Refer to __[`OVMSAdapter`](adapters/ovms_adapter.md)__ to learn about running demos with OVMS.

For using OpenVINO Model Server Adapter you need to install the package with extra module:
```sh
pip install <omz_dir>/demos/common/python[ovms]
```

## Model API Pipelines

Model API Pipelines represent the high-level wrappers upon the input data and accessing model results management.
They perform the data submission for model inference, verification of inference status, whether the result is ready or not, and results accessing.

The `AsyncPipeline` is available, which handles the asynchronous execution of a single model.

## Ready-to-use Model API solutions

To apply Model API wrappers in custom applications, learn the provided example of common scenario of how to use Model API.

 In the example, the SSD architecture is used to predict bounding boxes on input image `"sample.png"`. The model execution is produced by `OpenvinoAdapter`, therefore we submit the path to the model's `xml` file.

Once the SSD model wrapper instance is created, we get the predictions by the model in one line: `ssd_model(input_data)` - the wrapper performs the preprocess method, synchronous inference on OpenVINO™ toolkit side and postprocess method.

```python
import cv2
# import model wrapper class
from openvino.model_zoo.model_api.models import SSD
# import inference adapter and helper for runtime setup
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core


# read input image using opencv
input_data = cv2.imread("sample.png")

# define the path to mobilenet-ssd model in IR format
model_path = "public/mobilenet-ssd/FP32/mobilenet-ssd.xml"

# create adapter for OpenVINO™ runtime, pass the model path
model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")

# create model API wrapper for SSD architecture
# preload=True loads the model on CPU inside the adapter
ssd_model = SSD(model_adapter, preload=True)

# apply input preprocessing, sync inference, model output postprocessing
results = ssd_model(input_data)
```

To study the complex scenarios, refer to Open Model Zoo Python* demos, where the asynchronous inference is applied.
