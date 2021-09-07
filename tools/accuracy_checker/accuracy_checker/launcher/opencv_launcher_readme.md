# How to configure OpenCV launcher

For enabling OpenCV launcher you need to add `framework: opencv` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer (`cpu`, `gpu`, `gpu_fp16` etc.).
* `tags` - optional specifies which device bit type will be used for infer (`FP32` and `FP16`).
* `backend` - specifies which backend OpenCV's will be used for infer (`ocv` and `ie`).
* `model/weights` - path to configuration files with model and weights for your topology (`prototxt/caffemodel`, `xml/bin`, `pbtxt/pb` etc.) and preferably used in pairs .
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).

You also should specify all inputs for your model with their shapes to write inputs, using specific parameter: `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need to provide `value`, because it will be calculated in runtime. Format value is list with `N` elements of the form `[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `ORIG_IMAGE_INFO` - specific key for setting information about original image size before preprocessing.
    * `IGNORE_INPUT` - input which should be stayed empty during evaluation.
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
  * `shape` - shape of input layer described as comma-separated of all dimensions size except batch size.

    Optionally you can determine `layout` in case when your model was trained with non-standard data layout (For OpenCV default layout is `NCHW`) and `precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

OpenCV launcher config example:

```yml
launchers:
  - framework: opencv
    device: CPU
    backend: OCV
    model: path_to_model/alexnet.prototxt
    weights: path_to_weights/alexnet.caffemodel
    inputs:
      - name: 'input'
        type: INPUT
        shape: 3, 32, 32
    adapter: classification
```
