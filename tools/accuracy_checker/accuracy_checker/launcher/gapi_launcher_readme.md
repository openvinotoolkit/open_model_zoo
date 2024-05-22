# How to configure G-API launcher

G-API launcher runs model using graph-based evaluation approach proposed by [OpenCV G-API](https://docs.opencv.org/4.5.2/d0/d1e/gapi.html) module and OpenVINO as inference backend.

For enabling G-API launcher you need to add `framework: g-api` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer. Supported: `CPU`, `GPU`, `FPGA`,
    Heterogeneous plugin as `HETERO:target_device,fallback_device` and Multi device plugin as `MULTI:target_device1,target_device2`.

    It is possible to specify one or more devices via `-td, --target devices` command line argument. Target device will be selected from command line (in case when several devices provided, evaluations will be run one by one with all specified devices).
* `model` - path to xml file with model for your topology or compiled executable network.
* `weights` - path to bin file with weights for your topology (Optional, the argument can be omitted if bin file stored in the same directory with model xml or if you use compiled blob).
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).
* `outputs` - list of model output names.
You also should specify all inputs for your model with their shapes to write inputs, using specific parameter: `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need to provide `value`, because it will be calculated in runtime. Format value is list with `N` elements of the form `[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `ORIG_IMAGE_INFO` - specific key for setting information about original image size before preprocessing.
    * `PROCESSED_IMAGE_INFO` - specific key for setting information about input size after preprocessing.
    * `IGNORE_INPUT` - input which should be stayed empty during evaluation.
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
  * `shape` - shape of input layer described as comma-separated of all dimensions size except batch size.

    Optionally you can determine `layout` in case when your model was trained with non-standard data layout (For G-API default layout is `NHWC`) and `precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

G-API launcher config example:

```yml
launchers:
  - framework: g-api
    device: CPU
    model: path_to_model/densenet-121-tf.xml
    weights: path_to_weights/densenet-121-tf.bin
    inputs:
      - name: 'data'
        type: INPUT
        shape: 3, 227, 227
    outputs:
      - prob
    adapter: classification
```
