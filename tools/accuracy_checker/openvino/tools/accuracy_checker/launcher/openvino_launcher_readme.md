# How to configure OpenVINO™ launcher

OpenVINO™ launcher is one of the supported wrappers for easily launching models within Accuracy Checker tool. This launcher uses OpenVINO™ with support API 2.0 as inference backend and accepts for executing networks in OpenVINO™ supported formats.

**Note**: the tool by default executes OpenVINO™ with API2.0 if it is available. In order to run model with API1.0 you need specify `--use_new_api False` parametrs, see [this section](#deprecated-how-to-configure-openvino-api-10-launcher) for details how to setup Accuracy Cheker launcher for API 1.0 and [migration instruction](#migrate-accuracy-checker-configuration-from-openvino-api-10-to-20) for configs.

For enabling OpenVINO™ launcher you need to add `framework: openvino` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer. Supported: `CPU`, `GPU`, `GNA`,
    Heterogeneous plugin as `HETERO:target_device,fallback_device` and Multi device plugin as `MULTI:target_device1,target_device2`.

    It is possible to specify one or more devices via `-td, --target devices` command line argument. Target device will be selected from command line (in case when several devices provided, evaluations will be run one by one with all specified devices).
* `model` - path to model for your topology or compiled executable network. You also can provide path to directory with model for automatic model search inside directory and help to find it specifying `--model_type`. Supported model types: `xml`, `onnx`, `paddle`, `tf`, `blob`.
* `weights` - path to bin file with weights for your topology (Optional, the argument can be omitted if bin file stored in the same directory with model xml or if you use compiled blob).
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).

Launcher understands which batch size will be used from model intermediate representation (IR). If you want to use batch for infer, please, provide model with required batch

* `allow_reshape_input` - parameter, which allows to reshape input layer to data shape (default value is False).
* `reset_memory_state` - parameter, which allows resetting internal infer request memory states after inference. State control essential for recurrent networks. (Optional, default is `False`).


Device config contains device specific options which should be set to Inference Engine. For setting device specific flags, you are able to use `-dc` or `--device_config` command line option and provide path to YML file or specify device config directly to Accuracy Checker config file with key `device_config` in the launchers section. Device config should be represented as dictionary of one of two types:
1. keys are plugin configuration keys and values are their values respectively. In this way configuration will be applied to current running device.
2. keys are supported devices and values are plugin configuration for each device. Plugin configuration represented as dictionary where keys are plugin specific configuration keys and values are their values respectively.

Each supported device has own set of supported configuration parameters which can be found on device page in [Inference Engine development guide](https://docs.openvino.ai/2023.0/_docs_IE_DG_supported_plugins_Supported_Devices.html)

**Note:** Since OpenVINO 2020.4 on platforms with native bfloat16 support models will be executed on this precision by default. For disabling this behaviour, you need to use device_config with following configuration:
```yml
CPU:
   ENFORCE_BF16: "NO"
```
Device config example can be found at `<omz_dir>/tools/accuracy_checker/sample/disable_bfloat16_device_config.yml`.

Beside that, you can launch model in `async_mode`, enable this option and optionally provide the number of infer requests (`num_requests`), which will be used in evaluation process. By default, if `num_requests` not provided or used value `AUTO`, automatic number request assignment for specific device will be performed
For multi device configuration async mode used always. You can provide number requests for each device as part device specification: `MULTI:device_1(num_req_1),device_2(num_req_2)` or in `num_requests` config section (for this case comma-separated list of integer numbers or one value if number requests for all devices equal can be used).

**Note:** not all models support async execution, in cases when evaluation can not be run in async, the inference will be switched to sync.

## Specifying model inputs in config

In case when you model has several inputs you should provide list of input layers in launcher config section using key `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need to provide `value`, because it will be calculated in runtime. Format value is list with `N` elements of the form `[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `ORIG_IMAGE_INFO` - specific key for setting information about original image size before preprocessing.
    * `PROCESSED_IMAGE_INFO` - specific key for setting information about input size after preprocessing.
    * `SCALE_FACTOR` - specific key for setting information about image scale factor defined as `[SCALE_Y, SCALE_X]`, where `SCALE_Y` = `<resized_image_height>/<original_image_height`, `SCALE_X` = `<resized_image_width> / <original_image_width>`
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
    * `LSTM_INPUT` - input which should be filled by hidden state from previous iteration. The hidden state layer name should be provided via `value` parameter.
    * `IGNORE_INPUT` - input which should be stayed empty during evaluation.

    Optionally you can determine `shape` of input (by default OpenVINO™ launcher uses info given from network, this option allows override default. It required for running ONNX\* models with dynamic input shape on devices, where dynamic shape is not supported), `layout` in case when your model was trained with non-standard data layout (For OpenVINO™ launcher default layout is `NCHW`)
    and `precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

## Configuration example

OpenVINO™ launcher config example:

```yml
launchers:
  - framework: openvino
    device: CPU
    model: path_to_model/alexnet.xml
    weights: path_to_weights/alexnet.bin
    adapter: classification
```

# [DEPRECATED] How to configure OpenVINO™ API 1.0 launcher

OpenVINO™ API 1.0 launcher is one of the supported wrappers for easily launching models within Accuracy Checker tool. This launcher uses OpenVINO™ API 1.0 as inference backend and accepts for executing networks in OpenVINO™ supported formats.
**Note:** Since OpenVINO™ 2022.1, API used as base for this launcher is deprecated
For enabling OpenVINO™ API 1.0 launcher you need to add `framework: dlsdk` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer. Supported: `CPU`, `GPU`, `GNA`, `MYRIAD`, `HDDL`,
    Heterogeneous plugin as `HETERO:target_device,fallback_device` and Multi device plugin as `MULTI:target_device1,target_device2`.

    If you have several devices in your machine, you are able to provide specific device id in such way: `<DEVICE>.<DEVICE_ID>` (e.g. `MYRIAD.1.2-ma2480`)

    It is possible to specify one or more devices via `-td, --target devices` command line argument. Target device will be selected from command line (in case when several devices provided, evaluations will be run one by one with all specified devices).
* `model` - path to model for your topology or compiled executable network. You also can provide path to directory with model for automatic model search inside directory and help to find it specifying `--model_type`. Supported model types: `xml`, `onnx`, `blob`.
* `weights` - path to bin file with weights for your topology (Optional, the argument can be omitted if bin file stored in the same directory with model xml or if you use compiled blob).
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here](../adapters/README.md).

Launcher understands which batch size will be used from model intermediate representation (IR). If you want to use batch for infer, please, provide model with required batch.

* `allow_reshape_input` - parameter, which allows to reshape input layer to data shape (default value is False).
* `reset_memory_state` - parameter, which allows resetting internal infer request memory states after inference. State control essential for recurrent networks. (Optional, default is `False`).


Device config contains device specific options which should be set to Inference Engine. For setting device specific flags, you are able to use `-dc` or `--device_config` command line option and provide path to YML file or specify device config directly to Accuracy Checker config file with key `device_config` in the launchers section. Device config should be represented as dictionary of one of two types:
1. keys are plugin configuration keys and values are their values respectively. In this way configuration will be applied to current running device.
2. keys are supported devices and values are plugin configuration for each device. Plugin configuration represented as dictionary where keys are plugin specific configuration keys and values are their values respectively.

Each supported device has own set of supported configuration parameters which can be found on device page in [Inference Engine development guide](https://docs.openvino.ai/2023.0/_docs_IE_DG_supported_plugins_Supported_Devices.html)

**Note:** Since OpenVINO 2020.4 on platforms with native bfloat16 support models will be executed on this precision by default. For disabling this behaviour, you need to use device_config with following configuration:
```yml
CPU:
   ENFORCE_BF16: "NO"
```
Device config example can be found at `<omz_dir>/tools/accuracy_checker/sample/disable_bfloat16_device_config.yml`.

Beside that, you can launch model in `async_mode`, enable this option and optionally provide the number of infer requests (`num_requests`), which will be used in evaluation process. By default, if `num_requests` not provided or used value `AUTO`, automatic number request assignment for specific device will be performed
For multi device configuration async mode used always. You can provide number requests for each device as part device specification: `MULTI:device_1(num_req_1),device_2(num_req_2)` or in `num_requests` config section (for this case comma-separated list of integer numbers or one value if number requests for all devices equal can be used).

**Note:** not all models support async execution, in cases when evaluation can not be run in async, the inference will be switched to sync.

## Specifying model inputs in config

In case when you model has several inputs you should provide list of input layers in launcher config section using key `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need to provide `value`, because it will be calculated in runtime. Format value is list with `N` elements of the form `[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `ORIG_IMAGE_INFO` - specific key for setting information about original image size before preprocessing.
    * `PROCESSED_IMAGE_INFO` - specific key for setting information about input size after preprocessing.
    * `SCALE_FACTOR` - specific key for setting information about image scale factor defined as `[SCALE_Y, SCALE_X]`, where `SCALE_Y` = `<resized_image_height>/<original_image_height`, `SCALE_X` = `<resized_image_width> / <original_image_width>`
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
    * `LSTM_INPUT` - input which should be filled by hidden state from previous iteration. The hidden state layer name should be provided via `value` parameter.
    * `IGNORE_INPUT` - input which should be stayed empty during evaluation.

    Optionally you can determine `shape` of input (by default OpenVINO™ launcher uses info given from network, this option allows override default. It required for running ONNX\* models with dynamic input shape on devices, where dynamic shape is not supported), `layout` in case when your model was trained with non-standard data layout (For OpenVINO™ launcher default layout is `NCHW`)
    and `precision` (Supported precisions: `FP32` - float, `FP16` - signed shot, `U8`  - unsigned char, `U16` - unsigned short int, `I8` - signed char, `I16` - short int, `I32` - int, `I64` - long int).

## Configuration example

OpenVINO™ launcher config example:

```yml
launchers:
  - framework: dlsdk
    device: CPU
    model: path_to_model/alexnet.xml
    weights: path_to_weights/alexnet.bin
    adapter: classification
```

# Migrate Accuracy Checker configuration from OpenVINO™ API 1.0 to 2.0

Both `openvino`   and `dlsdk` launchers  supports execution models in OpenVINO™ IR v10 and OpenVINO™ IR v11 formats, but in some cases for successful launching, you need adjust configuration previously used with API 1.0:
1. Align output names in `adapter` according to the launcher expected if they are specified.
2. If model inputs layout is different for APIs (e.g. for Tensorflow\* converted IR v11 with NHWC input layout, no conversion to NCHW with API 2.0 by default) and it specified in `inputs` section of launcher config (or override it using command line parameter `--layout`).
3. Change `framework` field is optional, you can switch between launchers just using `--use_new_api` parameter (if it is set to `True`, then `dlsdk` will be switched to `openvino` if additional configuration for `openvino` launcher is not provided in the config. Setting parameter to `False` works in opposite manner). Command line argument `--target_frameworks`(`-tf)` behaviour is also aligned with this logic for filtering configuration.
