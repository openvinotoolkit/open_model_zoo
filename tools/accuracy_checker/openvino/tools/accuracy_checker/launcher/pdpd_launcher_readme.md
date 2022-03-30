# How to configure PaddlePaddle launcher

PaddlePaddle launcher is one of the supported wrappers for easily launching models within Accuracy Checker tool. This launcher allows to execute models in PaddlePaddle framework.

For enabling PaddlePaddle launcher you need to add `framework: paddle_paddle` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer (`cpu`, `gpu` and so on). Optional, cpu used as default.
* `model`- path to the network file with inference model.
* `params` - path to network parameters file.
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).

# Specifying model inputs in config.

In case when you model has several inputs you should provide list of input layers in launcher config section using key `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need to provide `value`, because it will be calculated in runtime. Format value is list with `N` elements of the form `[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `ORIG_IMAGE_INFO` - specific key for setting information about original image size before preprocessing.
    * `PROCESSED_IMAGE_INFO` - specific key for setting information about input size after preprocessing.
    * `SCALE_FACTOR` - specific key for setting information about image scale factor defined as `[SCALE_Y, SCALE_X]`, where `SCALE_Y` = `<resized_image_height>/<original_image_height`, `SCALE_X` = `<resized_image_width> / <original_image_width>`
    * `IGNORE_INPUT` - input which should be stayed empty during evaluation.
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.

    Optionally you can determine `shape` of input (actually does not used, PaddlePaddle launcher uses info given from network),`layout` in case when your model was trained with non-standard data layout (For PaddlePaddle launcher default layout is `NCHW`)
    and `precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

PaddlePaddle launcher config example:

```yml
launchers:
  - framework: paddle_paddle
    device: CPU
    model: path_to_model/ResNet18/model
    params: path_to_model/ResNet18/params
    adapter: classification
```
