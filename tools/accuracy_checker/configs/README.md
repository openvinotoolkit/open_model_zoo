# How to use predefined configuration files

**Note**: Since 2022.1 all provided Open Model Zoo configuration files switched to work with OpenVINO™ API 2.0. For launching models with API 1.0, additional actions may be required, see [config migration guide](../accuracy_checker/launcher/openvino_launcher_readme.md#migrate-accuracy-checker-configuration-from-openvino-api-1-0-to-2-0).


## Structure

Configuration file declares validation process. Every model has to have entry in `models` list. Each entry has to contain distinct `name`, `launchers` and `datasets` sections.

Example:

```yaml
models:
  - name: model_name

    launchers:
      - framework: openvino
        adapter: adapter_name

    datasets:
      - name: dataset_name
```

Also there are composite models which consist of several parts (models) and the accuracy measurement requires building the pipeline from these parts. Thus, the evaluation is performed by sequentially executing a set of models and impossible to evaluate them independently. Each composite model has to have entry in `evaluations` list. Each entry should contain distinct `name`, `module` and `module_config`. `module_config` has to consist of `network_info`,`launchers` and `datasets` fields. Custom evaluators are used for such models. More information about defining and using your own evaluator or an existing one can be found in [Custom Evaluators Guide](../accuracy_checker/evaluators/custom_evaluators/README.md)

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
        - framework: openvino

      datasets:
        - name: dataset_name
```

## Location

Predefined configuration file `accuracy-check.yml` for each Open Model Zoo model can be found in the model directory.

`<model_name>.yml` file, which is located in current `configs` folder, is a link to `accuracy-check.yml` for `<model_name>` model.

Example:

`<omz_dir>/tools/accuracy_checker/configs/densenet-121-tf.yml` is a link for configuration file `<omz_dir>/models/public/densenet-121-tf/accuracy-check.yml` for [densenet-121-tf](../../../models/public/densenet-121-tf/README.md) model.

## Options

1. To run configuration specify the path to the required configuration file to `-c, --config` command line.
2. Configuration files don't contain paths to used models and weights. The model and weights are searched automatically by name of model in path specified in `-m, --models` command line option.
3. There is global configuration file with dataset conversion parameters which is used to avoiding duplication. Global definitions will be merged with evaluation config in the runtime by dataset name. You can use global_definitions to specify path to this file via command line arguments `-d, --definitions`. In order, if you want use definitions file in quantization via Post Training Optimization Toolkit, you should use environment variable `DEFINITIONS_FILE` for specifying path to definitions.
4. The path relative to which the `data_source` is specified can be provided via  `-s, --source` command line. If you want to evaluate models using well-known datasets, you need to organize folders with validation datasets in a certain way. More detailed information about dataset preparation you can find in [Dataset Preparation Guide](../../../data/datasets.md). In order, if you want use data source in quantization via Post Training Optimization Toolkit, you should use environment variable `DATA_DIR` for specifying path to root of directories with datasets.
5. The path relative to which the  `annotation` and `dataset_meta` are specified can be provided via `-a, --annotations` command line.   Annotation and dataset_meta (if required) will be stored to this directory after annotation conversion step if they do not exist and can be used for the next running to skip annotation conversion. Detailed information about annotation conversion you can find in [Annotation Conversion Guide](../accuracy_checker/annotation_converters/README.md).
6. Some models can have additional files for evaluation (for example, vocabulary files for NLP models), generally, named as model attributes. The relative paths to model specific attributes(vocabulary files, merges files, etc.) can be provided in the configuration file, if it is required. The path prefix for them should be passed through `--model_attributes` command line option (usually, it is the model directory).
7. To specify devices for infer use `-td, --target_devices` command line option. Several devices should be separated by spaces (e.g. -td CPU GPU).
8. Optionally, if several frameworks are provided in the configuration file, you can specify inference framework for evaluation using `-tf, --target_framework` command line option. Otherwise, if the option is not provided evaluation will be launched with all frameworks mentioned in the configuration file.

## Example of usage

See how to evaluate model with using predefined configuration file for [densenet-121-tf](../../../models/public/densenet-121-tf/README.md) model.

- `OMZ_ROOT` - root of Open Model Zoo project
- `DATASET_DIR` - root directory with dataset
- `MODEL_DIR` - root directory with model

1. First of all, you need to prepare dataset according to [Dataset Preparation Guide](../../../data/datasets.md)
2. Download original model files from online source using [Model Downloader](../../../tools/model_tools/README.md)
    ```sh
    omz_downloader --name densenet-121-tf --output_dir MODEL_DIR
    ```
3. Convert model in the OpenVINO IR format using Model Optimizer via [Model Converter](../../../tools/model_tools/README.md)
    ```sh
    omz_converter --name densenet-121-tf --download_dir MODEL_DIR
    ```
4. Run evaluation for model in FP32 precision using [Accuracy Checker](../README.md)
    ```sh
    accuracy_check -c OMZ_ROOT/models/public/densenet-121-tf/accuracy-check.yml -s DATASET_DIR -m MODEL_DIR/public/densenet-121-tf/FP32 -d OMZ_ROOT/tools/accuracy_checker/dataset_definitions.yml -td CPU
    ```
    Similarly, you can run evaluation for model in FP16 precision
    ```sh
    accuracy_check -c OMZ_ROOT/models/public/densenet-121-tf/accuracy-check.yml -s DATASET_DIR -m MODEL_DIR/public/densenet-121-tf/FP16 -d OMZ_ROOT/tools/accuracy_checker/dataset_definitions.yml -td GPU
    ```
