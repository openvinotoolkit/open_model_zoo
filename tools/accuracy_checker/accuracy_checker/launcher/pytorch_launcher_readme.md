# How to configure PyTorch launcher

PyTorch launcher is one of the supported wrappers for easily launching models within Accuracy Checker tool. This launcher allows to execute models using PyTorch\* framework as inference backend.

For enabling PyTorch launcher you need to add `framework: pytorch` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer (`cpu`, `cuda` and so on).
* `module`- PyTorch network module for loading.
* `checkpoint` - pre-trained model checkpoint (Optional).
* `python_path` - appendix for PYTHONPATH for making network module visible in current python environment (Optional).
* `module_args` - list of positional arguments for network module (Optional).
* `module_kwargs` - dictionary (`key`: `value` where `key` is argument name, `value` is argument value) which represent network module keyword arguments.
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).
* `batch` - batch size for running model (Optional, default 1).

In turn if you model has several inputs you need to specify them in config, using specific parameter: `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need to provide `value`, because it will be calculated in runtime. Format value is list with `N` elements of the form `[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `ORIG_IMAGE_INFO` - specific key for setting information about original image size before preprocessing.
    * `IGNORE_INPUT` - input which should be stayed empty during evaluation.
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
  * `shape` - shape of input layer described as comma-separated of all dimensions size except batch size.

    Optionally you can determine `layout` in case when your model was trained with non-standard data layout (For PyTorch default layout is `NCHW`) and`precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

If you model has several outputs you also need to specify their names in config for ability to get their values in adapter using option `output_names`.

PyTorch launcher config example (demonstrates how to run AlexNet model from [torchvision](https://pytorch.org/vision/stable/models.html)):

```yml
launchers:
  - framework: pytorch
    device: CPU
    module: torchvision.models.alexnet

    module_kwargs:
      pretrained: True

    adapter: classification
```
