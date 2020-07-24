# How to configure OpenVINO™ launcher

For enabling OpenVINO™ launcher you need to add `framework: dlsdk` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer. Supported: `CPU`, `GPU`, `FPGA`, `MYRIAD`, `HDDL`,
Heterogeneous plugin as `HETERO:target_device,fallback_device` and Multi device plugin as `MULTI:target_device1,target_device2`.
If you have several MYRIAD devices in your machine, you are able to provide specific device id in such way: `MYRIAD.<DEVICE_ID>` (e.g. `MYRIAD.1.2-ma2480`)
It is possible to specify one or more devices via `-td, --target devices` command line argument. Target device will be selected from command line (in case when several devices provided, evaluations will be run one by one with all specified devices).
* `model` - path to xml file with model for your topology or compiled executable network.
* `weights` - path to bin file with weights for your topology (Optional, the argument can be omitted if bin file stored in the same directory with model xml or if you use compiled blob).

**Note:** 
   You can generate executable blob using [compile_tool](https://docs.openvinotoolkit.org/latest/_inference_engine_tools_compile_tool_README.html).
   Before evaluation executable blob, please make sure that selected device support it.


launcher may optionally provide model parameters in source framework format which will be converted to Inference Engine IR using Model Optimizer.
If you want to use Model Optimizer for model conversion, please view [Model Optimizer Developer Guide](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).
You can provide:

* `caffe_model` and `caffe_weights` for Caffe model and weights (*.prototxt and *.caffemodel).
* `tf_model` for TensorFlow model (*.pb, *.pb.frozen, *.pbtxt).
* `tf_meta` for TensorFlow MetaGraph (*.meta).
* `mxnet_weights` for MXNet params (*.params).
* `onnx_model` for ONNX model (*.onnx). You also able to pass your ONNX model directly using `model` option if you do not need Model Optimizer conversion step.
* `kaldi_model` for Kaldi model (*.nnet).

In case when you want to determine additional parameters for model conversion (data_type, input_shape and so on), you can use `mo_params` for arguments with values and `mo_flags` for positional arguments like `legacy_mxnet_model` .
Full list of supported parameters you can find in Model Optimizer Developer Guide.

Model will be converted before every evaluation. 
You can provide `converted_model_dir` for saving converted model in specific folder, otherwise, converted models will be saved in path provided via `-C` command line argument or source model directory.

* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).

Launcher understands which batch size will be used from model intermediate representation (IR). If you want to use batch for infer, please, provide model with required batch or convert it using specific parameter in `mo_params`.

* `allow_reshape_input` - parameter, which allows to reshape input layer to data shape (default value is False).

Additionally you can provide device specific parameters:

* `cpu_extensions` (path to extension file with custom layers for cpu). You can also use special key `AUTO` for automatic search cpu extensions library in the provided as command line argument directory (option `-e, --extensions`)
* `gpu_extensions` (path to extension *.xml file with OpenCL kernel description for gpu).
* `bitstream` for running on FPGA.

For setting device specific flags, you are able to use `-dc` or `--device_config` command line option. Device config should be represented as YML file with dictionary, where keys are plugin configuration keys and values are their values respectively.
Each supported device has own set of supported configuration parameters which can be found on device page in [Inference Engine development guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html)

**Note:** Since OpenVINO 2020.4 on platforms with native bfloat16 support models will be executed on this precision by default. For disabling this behaviour, you need to use device_config with following configuration:
```yml
ENFORCE_BF16: "NO"
```
Device config example can be found <a href="https://github.com/opencv/open_model_zoo/blob/develop/tools/accuracy_checker/sample/disable_bfloat16_device_config.yml">here</a>()

Beside that, you can launch model in `async_mode`, enable this option and optionally provide the number of infer requests (`num_requests`), which will be used in evaluation process. By default, if `num_requests` not provided or used value `AUTO`, automatic number request assignment for specific device will be performed
For multi device configuration async mode used always. You can provide number requests for each device as part device specification: `MULTI:device_1(num_req_1),device_2(num_req_2)` or in `num_requests` config section (for this case comma-separated list of integer numbers or one value if number requests for all devices equal can be used).

**Note:** not all models support async execution, in cases when evaluation can not be run in async, the inference will be switched to sync.
## Specifying model inputs in config.

In case when you model has several inputs you should provide list of input layers in launcher config section using key `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need provide `value`, because it will be calculated in runtime. Format value is `Nx[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
    * `LSTM_INPUT` - input which should be filled by hidden state from previous iteration. The hidden state layer name should be provided via `value` parameter.
    Optionally you can determine `shape` of input (actually does not used, DLSDK launcher uses info given from network), `layout` in case when your model was trained with non-standard data layout (For DLSDK default layout is `NCHW`)
    and `precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

OpenVINO™ launcher config example:

```yml
launchers:
  - framework: dlsdk
    device: HETERO:FPGA,CPU
    caffe_model: path_to_model/alexnet.prototxt
    caffe_weights: path_to_weights/alexnet.caffemodel
    adapter: classification
    mo_params:
      batch: 4
    mo_flags:
      - reverse_input_channels
    cpu_extensions: cpu_extentions_avx512.so
```
