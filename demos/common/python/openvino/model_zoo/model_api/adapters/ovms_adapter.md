# OpenVINO Model Server Adapter

The `OVMSAdapter` implements `ModelAdapter` interface. The `OVMSAdapter` makes it possible to use Model API with models hosted in OpenVINO Model Server.

## Prerequisites

`OVMSAdapter` enables inference via gRPC calls to OpenVINO Model Server, so in order to use it you need two things:
- OpenVINO Model Server that serves your model
- [`ovmsclient`](https://pypi.org/project/ovmsclient/) package installed to enable communication with the model server

### Deploy OpenVINO Model Server

Model Server is distributed as a docker image and it's available in DockerHub, so you can use it with `docker run` command. See [model server documentation](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md) to learn how to deploy OpenVINO optimized models with OpenVINO Model Server.

### Install ovmsclient

`ovmsclient` package is distributed on PyPi, so the easiest way to install it is via:

```
pip3 install ovmsclient
```

## Model configuration

When using OpenVINO Model Server model cannot be directly accessed from the client application (like OMZ demos). Therefore any configuration must be done on model server side.

### Input reshaping

For some use cases you may want your model to reshape to match input of certain size. In that case, you should provide `--shape auto` parameter to model server startup command. With that option, model server will reshape model input on demand to match the input data.

### Inference options

It's possible to configure inference related options for the model in OpenVINO Model Server with options:
- `--target_device` - name of the device to load the model to
- `--nireq` - number of InferRequests
- `--plugin_config` - configuration of the device plugin

See [model server configuration parameters](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md#configuration-parameters) for more details.

### Example OVMS startup command
```
docker run -d --rm -v /home/user/models:/models -p 9000:9000 openvino/model_server:latest --model_path /models/model1 --model_name model1 --port 9000 --shape auto --nireq 32 --target_device CPU --plugin_config "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_AUTO\"}"
```

> **Note**: In demos, while using `--adapter ovms`, inference options like: `-nireq`, `-nstreams` `-nthreads` as well as device specification with `-d` will be ignored.

## Running demos with OVMSAdapter

To run the demo with model served in OpenVINO Model Server, you would have to provide `--adapter ovms` option and modify `-m` parameter to indicate model inference service instead of the model files. Model parameter for `OVMSAdapter` follows this schema:

```<service_address>/models/<model_name>[:<model_version>]```

- `<service_address>` - OVMS gRPC service address in form `<address>:<port>`
- `<model_name>` - name of the target model (the one specified by `model_name` parameter in the model server startup command)
- `<model_version>` *(optional)* - version of the target model (default: latest)

 Assuming that model server runs on the same machine as the demo, exposes gRPC service on port 9000 and serves model called `model1`, the value of `-m` parameter would be:

- `localhost:9000/models/model1` - requesting latest model version
- `localhost:9000/models/model1:2` - requesting model version number 2

## See Also

* [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
