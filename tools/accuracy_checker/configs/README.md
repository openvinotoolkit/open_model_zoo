# How to use configuration files

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

## Location

Configuration file `accuracy-check.yml` is located in the `models` folder with model to which it belongs.

`model_name.yml` file, which is located in current `configs` folder, is a link to `accuracy-check.yml` for `model_name` model.

## Options

1. To run configuration specify the path to the required configuration file to `-c, --config` command line.
2. Configuration files don't contain paths to used models and weights. The model and weights are searched automatically by name of model in path specified in `-m, --models` command line option.
3. The path relative to which the `data_source` is specified can be provided via  `-s, --source` command line.
4. There is global configuration file with dataset conversion parameters which is used to avoiding duplication. Global definitions will be merged with evaluation config in the runtime by dataset name. You can use global_definitions to specify path to this file via command line arguments `-d, --definitions`.
5. Also relative paths to additional models attributes can be specified in the configuration file. The directory relative to which these attributes will be searched can be passed through `--model_attributes` command line option.
6. To specify devices for infer use `-td, --target_devices` command line option. Several devices should be separated by spaces.
