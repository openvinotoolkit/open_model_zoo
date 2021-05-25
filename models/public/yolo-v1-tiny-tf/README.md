# yolo-v1-tiny-tf

## Use Case and High-Level Description

YOLO v1 Tiny is a real-time object detection model from TensorFlow.js\* framework. This model was pre-trained on VOC dataset with 20 classes.

## Conversion

1. Install additional dependencies:
   ```
   h5py
   keras
   tensorflowjs
   ```
2. Download model from [here](https://github.com/shaqian/tfjs-yolo-demo/tree/master/dist/model/v1tiny) (tested on `aa4354c` commit).
3. Convert model to Keras\* format using `tensorflowjs_converter` script, e.g.:
   ```
   tensorflowjs_converter --input_format tfjs_layers_model --output_format keras <model_in>.json <model_out>.h5
   ```
4. Convert the produced model to protobuf format.

   1. Get conversion script from [repository](https://github.com/amir-abdi/keras_to_tensorflow):
   ```
   git clone https://github.com/amir-abdi/keras_to_tensorflow
   ```
   2. (Optional) Checkout the commit that the conversion was tested on:
   ```
   git checkout c841508a88faa5aa1ffc7a4947c3809ea4ec1228
   ```
   3. Apply `keras_to_tensorflow.py.patch`:
   ```
   git apply keras_to_tensorflow.py.patch
   ```
   4. Run script:
   ```
   python keras_to_tensorflow.py --input_model=<model_in>.h5 --output_model=<model_out>.pb
   ```

## Specification

| Metric            | Value           |
|-------------------|-----------------|
| Type              | Detection       |
| GFLOPs            | 6.988           |
| MParams           | 15.858          |
| Source framework  | TensorFlow.js\* |

## Accuracy

Accuracy metric obtained on test data from VOC2007 dataset for converted model.

| Metric | Value  |
| ------ | -------|
| mAP    | 54.79% |

## Input

### Original model

Image, name - `input_1`, shape - `1, 416, 416, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `input_1`, shape - `1, 3, 416, 416`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

The array of detection summary info, name - `conv2d_9/BiasAdd`,  shape - `1, 13, 13, 125`, format is `B, Cx, Cy, N*25`, where:

- `B` - batch size
- `N` - number of detection boxes for cell
- `Cx`, `Cy` - cell index

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_20`], where:

- (`x`, `y`) - raw coordinates of box center, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get coordinates relative to the cell
- `h`, `w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get height and width values relative to cell
- `box_score` - confidence of detection box, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0, 1] range
- `class_no_1`, ..., `class_no_20` - probability distribution over the classes in logits format, apply [softmax function](https://en.wikipedia.org/wiki/Softmax_function) and multiply by obtained confidence value to get confidence of each class

Mapping to class names provided by `<omz_dir>/data/dataset_classes/voc_20cl.txt` file.

The anchor values are `1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52`.

### Converted model

The array of detection summary info, name - `conv2d_9/BiasAdd/YoloRegion`,  shape - `1, 21125`, which could be reshaped to `1, 125, 13, 13`, format is `B, N*25, Cx, Cy`, where:

- `B` - batch size
- `N` - number of detection boxes for cell
- `Cx`, `Cy` - cell index

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_20`], where:

- (`x`, `y`) - coordinates of box center relative to the cell
- `h`, `w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply with corresponding anchors to get height and width values relative to the cell
- `box_score` - confidence of detection box in [0, 1] range
- `class_no_1`, ..., `class_no_20` - probability distribution over the classes in the [0, 1] range, multiply by confidence value to get confidence of each class

Mapping to class names provided by `<omz_dir>/data/dataset_classes/voc_20cl.txt` file.

The anchor values are `1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52`.

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/shaqian/tfjs-yolo/master/LICENSE):

```
Copyright (c) 2018 Qian Sha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
```
