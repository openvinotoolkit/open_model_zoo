
## OpenVINO Model Server Adapter

The `OvmsAdapter` implements `ModelAdapter` interface which makes it possible to use Model API with models hosted in OpenVINO Model Server.


## Install OpenVINO Model Server

Model Server is distributed as a docker image, so you can pull it from Docker Hub with:

```
docker pull openvino/model_server
```

See [model server documentation](https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md) to learn how to deploy OpenVINO optimized models with OpenVINO Model Server.

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
docker run -d --rm -v <models_repository>:/models -p 9000:9000 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --shape auto --nireq 32 --target_device CPU --plugin_config "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_AUTO\"}"
```

> **Note**: In demos, while using `--adapter ovms`, inference options like: `-nireq`, `-nstreams` `-nthreads` as well as device specification with `-d` will be ignored.

## Running demos with OvmsAdapter

To run the demo with model served in OpenVINO Model Server, you would have to provide `--adapter ovms` option and modify `-m` parameter to indicate model inference service instead of the model files. Model parameter for `OvmsAdapter` follows this schema:

```<service_address>/models/<model_name>[:<model_option>]```

- `<service_address>` - OVMS gRPC service address in form `<address>:<port>`
- `<model_name>` - name of the target model (the one specified by `model_name` parameter in the model server startup command)
- `<model_version>` *(optional)* - version of the target model (default: latest)

 Assuming that model server is running on the same machine as the demo, expose gRPC service on port 9000 and serves model called `my_model`, the value of `-m` parameter would be:

`localhost:9000/models/my_model`

## See Also

* [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
