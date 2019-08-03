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
