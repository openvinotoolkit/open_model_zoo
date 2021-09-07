# How to configure Caffe launcher

Caffe launcher is one of the supported wrappers for easily launching models within Accuracy Checker tool. This launcher allows to execute models using Caffe\* framework as inference backend.

For enabling Caffe launcher you need to add `framework: caffe` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer (`cpu`, `gpu_0` and so on).
* `model` - path to prototxt file with Caffe model for your topology. (Optional, if not provided the search of model will be performed)
* `weights` - path to caffemodel file with weights for your topology. (Optional, if not provided, the search of caffemodel will be performed in the same directory where prototxt located)
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).

You also can specify batch size for your model using `batch` and allow to reshape input layer to data shape, using specific parameter: `allow_reshape_input` (default value is False).

## Specifying model inputs in config.

In case when you model has several inputs you should provide list of input layers in launcher config section using key `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need to provide `value`, because it will be calculated in runtime. Format value is list with `N` elements of the form `[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `ORIG_IMAGE_INFO` - specific key for setting information about original image size before preprocessing.
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
    * `IGNORE_INPUT` - input which should be stayed empty during evaluation.

    Optionally you can determine `shape` of input (actually does not used, Caffe launcher uses info given from network), `layout` in case when your model was trained with non-standard data layout (For Caffe default layout is `NCHW`)
    and `precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

Caffe launcher config example:

```yml
launchers:
  - framework: caffe
    device: CPU
    model: path_to_model/alexnet.prototxt
    weights: path_to_weights/alexnet.caffemodel
    adapter: classification
    batch: 4
```
