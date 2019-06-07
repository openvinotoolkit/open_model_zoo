# How to configure Caffe launcher

For enabling Caffe launcher you need to add `framework: caffe` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer (`cpu`, `gpu_0` and so on).
* `model` - path to prototxt file with Caffe model for your topology.
* `weights` - path to caffemodel file with weights for your topology.
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here][adapters].

You also can specify batch size for your model using `batch` and allow to reshape input layer to data shape, using specific parameter: `allow_reshape_input` (default value is False).

## Specifying model inputs in config.

In case when you model has several inputs you should provide list of input layers in launcher config section using key `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need provide `value`, because it will be calculated in runtime. Format value is `Nx[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
    Optionally you can determine `shape` of input (actually does not used, Caffe launcher uses info given from network) and `layout` in case when your model was trained with non-standard data layout (For Caffe default layout is `NCHW`).
    
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

[adapters]: ../adapters/README.md
