# How to configure OpenVINO™ launcher

For enabling OpenVINO™ launcher you need to add `framework: dlsdk` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer. Supported: `CPU`, `GPU`, `FPGA`, `MYRIAD` and Heterogeneous plugin as `HETERO:target_device,fallback_device`.
* `model` - path to xml file with Caffe model for your topology.
* `weights` - path to bin file with weights for your topology.

launcher may optionally provide model parameters in source framework format which will be converted to Inference Engine IR using Model Optimizer.
If you want to use Model Optimizer for model conversion, please view [Model Optimizer Developer Guide][openvino-mo].
You can provide:

* `caffe_model` and `caffe_weights` for Caffe model and weights (*.prototxt and *.caffemodel).
* `tf_model` for TensorFlow model (*.pb, *.pb.frozen, *.pbtxt).
* `tf_meta` for TensorFlow MetaGraph (*.meta).
* `mxnet_weights` for MXNet params (*.params).
* `onnx_model` for ONNX model (*.onnx).
* `kaldi_model` for Kaldi model (*.nnet).

In case when you want to determine additional parameters for model conversion (data_type, input_shape and so on), you can use `mo_params` for arguments with values and `mo_flags` for positional arguments like `legacy_mxnet_model` .
Full list of supported parameters you can find in Model Optimizer Developer Guide.

Model will be converted before every evaluation. 
You can provide `converted_model_dir` for saving converted model in specific folder, otherwise, converted models will be saved in path provided via `-C` command line argument or source model directory.

* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here][adapters].

Launcher understands which batch size will be used from model intermediate representation (IR). If you want to use batch for infer, please, provide model with required batch or convert it using specific parameter in `mo_params`.

* `allow_reshape_input` - parameter, which allows to reshape input layer to data shape (default value is False).

Additionally you can provide device specific parameters:

* `cpu_extensions` (path to extension file with custom layers for cpu). You can also use special key `AUTO` for automatic search cpu extensions library in the provided as command line argument directory (option `-e, --extensions`)
* `gpu_extensions` (path to extension *.xml file with OpenCL kernel description for gpu).
* `bitstream` for running on FPGA.

Beside that, you can launch model in `async_mode`, enable this option and provide the number of infer requests (`num_requests`), which will be used in evaluation process

## Specifying model inputs in config.

In case when you model has several inputs you should provide list of input layers in launcher config section using key `inputs`.
Each input description should has following info:
  * `name` - input layer name in network
  * `type` - type of input values, it has impact on filling policy. Available options:
    * `CONST_INPUT` - input will be filled using constant provided in config. It also requires to provide `value`.
    * `IMAGE_INFO` - specific key for setting information about input shape to layer (used in Faster RCNN based topologies). You do not need provide `value`, because it will be calculated in runtime. Format value is `Nx[H, W, S]`, where `N` is batch size, `H` - original image height, `W` - original image width, `S` - scale of original image (default 1).
    * `INPUT` - network input for main data stream (e. g. images). If you have several data inputs, you should provide regular expression for identifier as `value` for specifying which one data should be provided in specific input.
    Optionally you can determine `shape` of input (actually does not used, DLSDK launcher uses info given from network) and `layout` in case when your model was trained with non-standard data layout (For DLSDK default layout is `NCHW`).

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

[adapters]: ../adapters/README.md
[openvino-mo]: https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer
