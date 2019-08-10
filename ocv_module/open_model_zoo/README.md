# Open Model Zoo

This is OpenCV module which let you have interactive Open Model Zoo in Python.

## Build

Add this module as an extra module

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_EXTRA_MODULES_PATH="~/open_model_zoo/ocv_module" \
      -DBUILD_LIST=python2,open_model_zoo ..
```


If you already have modules such opencv_contrib, you can combine it. In example,

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOPENCV_EXTRA_MODULES_PATH="~/open_model_zoo/ocv_module;~/opencv_contrib/modules" \
      -DBUILD_LIST=python2,open_model_zoo,aruco ..
```

## Usage

```python
import cv2 as cv
from cv2.open_model_zoo import squeezenet1_0

net = squeezenet1_0()
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
