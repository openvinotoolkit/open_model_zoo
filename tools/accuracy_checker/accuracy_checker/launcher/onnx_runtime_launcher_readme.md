# How to configure ONNX Runtime launcher

For enabling ONNX Runtime launcher you need to add `framework: onnx_runtime` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer (`cpu`, `gpu` and so on). Optional, cpu used as default or can depend on used executable provider.
* `model`- path to the network file in ONNX format.
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).
* `execution_providers` - list of execution providers for evaluation, e.g. <a href="https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/OpenVINO-ExecutionProvider.md">OpenVINO Execution Provider</a>. Default [`CPUExecutionProvider`] used. 

**Note: execution providers available only with newest versions of ONNXRuntime, if your installed version does not support such API, please update or does not specify this field.**


# Specifying model inputs in config.

In case when you model has several inputs you should provide list of input layers in launcher config section using key `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need provide `value`, because it will be calculated in runtime. Format value is `Nx[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
    Optionally you can determine `shape` of input (actually does not used, ONNX Runtime launcher uses info given from network),`layout` in case when your model was trained with non-standard data layout (For ONNX Runtime default layout is `NCHW`)
    and `precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

MXNet launcher config example:

```yml
launchers:
  - framework: onnx_runtime
    device: CPU
    model: path_to_model/alexnet.onnx
    adapter: classification
```
