# Open Model Zoo

This is OpenCV module which let you have interactive Open Model Zoo in Python.

## Build

Add this module as an extra module

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_EXTRA_MODULES_PATH="~/open_model_zoo/ocv_module" \
      -DBUILD_LIST=python3,open_model_zoo ..
```


If you already have modules such opencv_contrib, you can combine it. In example,

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_EXTRA_MODULES_PATH="~/open_model_zoo/ocv_module;~/opencv_contrib/modules" \
      -DBUILD_LIST=python3,open_model_zoo,aruco ..
```

### Build with Inference Engine support (to run OpenVINO models)

* Download and install Intel OpenVINO from https://software.seek.intel.com/openvino-toolkit

```bash
source /opt/intel/openvino/bin/setupvars.sh

cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_INF_ENGINE=ON \
      -DOPENCV_EXTRA_MODULES_PATH="~/open_model_zoo/ocv_module" \
      -DBUILD_LIST=highgui,imgcodecs,dnn,python3,open_model_zoo ..

make -j4
```

Override paths to newly built OpenCV:
```
export LD_LIBRARY_PATH=/path/to/opencv/build/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/opencv/build/lib/python3:$PYTHONPATH
```

## Usage

Every topology is a meta object which contains description, preprocessing parameters and URLs to download the model.
Creating an object you instantiate downloading of necessary files for specific model.

```python
import cv2 as cv
from cv2.open_model_zoo.topologies import squeezenet1_0

topology = squeezenet1_0()
```

To infer network you can use as just paths downloaded files:
```python
from openvino.inference_engine import IENetwork

xmlPath, binPath = topology.convertToIR()

net = IENetwork(xmlPath, binPath)
```

either one of included wrappers (in example, OpenCV's `dnn::ClassificationModel`):

```python
from cv2.open_model_zoo import DnnClassificationModel

net = DnnClassificationModel(topology)

img = cv.imread('example.jpg')

classId, confidence = net.classify(img)
```

## Algorithms

Some of networks may have pretty complicated pre- or post- processing procedures.
Another models can be combined to solve interesting problems. For these kind of
topologies you can use ready Algorithms. In example, to recognize text:

```python
from cv2.open_model_zoo import TextRecognitionPipeline

pipeline = TextRecognitionPipeline()

img = cv.imread('example.jpg')

rotatedRects, texts, confidences = pipeline.process(img)
```

## Tests

To run Python tests, build the module and type the following command:
```bash
export OPENCV_PYTEST_FILTER=test_omz
cd ~/open_model_zoo/ocv_module/open_model_zoo/misc/python/test
python3 ~/opencv/modules/python/test/test.py --repo ~/opencv -v
```

To run C++ tests, build OpenCV with tests enabled (`-DBUILD_TESTS=ON` and with `ts` module added to `-DBUILD_LIST`) then run
```
./bin/opencv_test_open_model_zoo
```

from OpenCV build folder.
