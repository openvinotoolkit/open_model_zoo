# How to use predefined configuration files

## Structure

Configuration file declares validation process. Every model has to have entry in `models` list. Each entry has to contain distinct `name`, `launchers` and `datasets` sections.

Example:

```yaml
models:
  - name: model_name

    launchers:
      - framework: dlsdk
        adapter: adapter_name

    datasets:
      - name: dataset_name
```

Also there are composite models, each of which has to have entry in `evaluations` list. Each entry has to contain distinct `name`, `module` and `module_config`. `module_config` has to consist of `network_info`,`launchers` and `datasets` fields.

Example:

```yaml
evaluations:
  - name: model_name
    module: name_of_class_with_custom_evaluators
    module_config:
      network_info:
        encoder: {}

        decoder:
          adapter: adapter_name

      launchers:
        - framework: dlsdk

      datasets:
        - name: dataset_name
```

## Location

Predefined configuration file `accuracy-check.yml` for each Open Model Zoo model can be found in the model directory.

`model_name.yml` file, which is located in current `configs` folder, is a link to `accuracy-check.yml` for `model_name` model.

Example:

[alexnet.yml](alexnet.yml) is a link for configuration file [accuracy-check.yml](../../../models/public/alexnet/accuracy-check.yml) for `alexnet` model.

## Options

1. To run configuration specify the path to the required configuration file to `-c, --config` command line.
2. Configuration files don't contain paths to used models and weights. The model and weights are searched automatically by name of model in path specified in `-m, --models` command line option.
3. The path relative to which the `data_source` is specified can be provided via  `-s, --source` command line. If you want to evaluate models using well-known datasets, you need to organize folders with validation datasets in a certain way. More detailed information about dataset preparation you can find in [Dataset Preparation Guide](../../../datasets.md).
4. The path relative to which the  `annotation` and `dataset_meta` are specified can be provided via `-a, --annotations` command line. Detailed information about annotation conversion you can find in [Annotation Conversion Guide](../accuracy_checker/annotation_converters/README.md).
5. There is global configuration file with dataset conversion parameters which is used to avoiding duplication. Global definitions will be merged with evaluation config in the runtime by dataset name. You can use global_definitions to specify path to this file via command line arguments `-d, --definitions`.
6. Also relative paths to additional models attributes(vocabulary files, merges files, etc.) can be specified in the configuration file. The directory relative to which these attributes will be searched can be passed through `--model_attributes` command line option.
7. To specify devices for infer use `-td, --target_devices` command line option. Several devices should be separated by spaces.

## Example of usage

See how to evaluate model with using predefined configuration file for [densenet-121-tf](../../../models/public/densenet-121-tf/densenet-121-tf.md) model.

- `OMZ_ROOT` - root of Open Model Zoo project
- `DATASET_DIR` - root directory with dataset
- `MODEL_DIR` - root directory with model

1. First of all, you need to prepare dataset according to [Dataset Preparation Guide](../../../datasets.md)
2. Download original model files from online source using [Model Downloader](../../../tools/downloader/README.md)
    ```sh
    ./downloader.py --name densenet-121-tf --output_dir MODEL_DIR
    ```
3. Convert model in the Inference Engine IR format using Model Optimizer via [Model Converter](../../../tools/downloader/README.md)
    ```sh
    ./converter.py --name densenet-121-tf --download_dir MODEL_DIR --mo my/openvino/path/model_optimizer/mo.py
    ```
4. Run evaluation for model in FP32 precision using [Accuracy Checker](../README.md)
    ```sh
    ./accuracy-check -c OMZ_ROOT/models/public/densenet-121-tf.yml -s DATASET_DIR -m MODEL_DIR/public/densenet-121-tf/FP32 -d OMZ_ROOT/tools/accuracy_checker/dataset_definitions.yml
    ```
    Similarly, you can run evaluation for model in FP16 precision
    ```sh
    ./accuracy-check -c OMZ_ROOT/models/public/densenet-121-tf.yml -s DATASET_DIR -m MODEL_DIR/public/densenet-121-tf/FP16 -d OMZ_ROOT/tools/accuracy_checker/dataset_definitions.yml
    ```
5. Also you can quantize full-precision models in the IR format into low-precision versions via [Model Quantizer](../../../tools/downloader/README.md)
    ```sh
    ./quantizer.py --name densenet-121-tf --dataset_dir DATASET_DIR --model_dir MODEL_DIR
    ```
   Run evaluation for quantized model:
    ```sh
    ./accuracy-check -c OMZ_ROOT/models/public/densenet-121-tf.yml -s DATASET_DIR -m MODEL_DIR/public/densenet-121-tf/FP16-INT8 -d OMZ_ROOT/tools/accuracy_checker/dataset_definitions.yml
    ```
