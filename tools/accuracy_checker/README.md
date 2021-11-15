# Deep Learning accuracy validation framework

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   omz_tools_accuracy_checker_adapters
   omz_tools_accuracy_checker_annotation_converters
   omz_tools_accuracy_checker_custom_evaluators
   omz_tools_accuracy_checker_data_readers
   omz_tools_accuracy_checker_caffe_launcher
   omz_tools_accuracy_checker_mxnet_launcher
   omz_tools_accuracy_checker_onnx_runtime_launcher
   omz_tools_accuracy_checker_opencv_launcher
   omz_tools_accuracy_checker_dlsdk_launcher
   omz_tools_accuracy_checker_pdpd_launcher
   omz_tools_accuracy_checker_pytorch_launcher
   omz_tools_accuracy_checker_tf2_launcher
   omz_tools_accuracy_checker_tf_lite_launcher
   omz_tools_accuracy_checker_tf_launcher
   omz_tools_accuracy_checker_configs
   omz_tools_accuracy_checker_metrics
   omz_tools_accuracy_checker_postprocessor
   omz_tools_accuracy_checker_preprocessor
   omz_tools_accuracy_checker_sample

@endsphinxdirective


The Accuracy Checker is an extensible, flexible and configurable Deep Learning accuracy validation framework. The tool has a modular structure and allows to reproduce validation pipeline and collect aggregated quality indicators for popular datasets both for networks in source frameworks and in the OpenVINO™ supported formats.

## Installation

> **TIP**: You can quick start with the Accuracy Checker inside the OpenVINO™ [Deep Learning Workbench](@ref 
> openvino_docs_get_started_get_started_dl_workbench) (DL Workbench).
> [DL Workbench](@ref workbench_docs_Workbench_DG_Introduction) is an OpenVINO™ UI that enables you to
> import a model, analyze its performance and accuracy, visualize the outputs, optimize and prepare the model for 
> deployment on various Intel® platforms.

### Prerequisites

Install prerequisites first:

#### 1. Python

**Accuracy Checker** uses **Python 3**. Install it first:

- [Python3](https://www.python.org/downloads/), [setuptools](https://pypi.org/project/setuptools/):

```bash
sudo apt-get install python3 python3-dev python3-setuptools python3-pip
```

Python\* setuptools and Python\* package manager (pip) install packages into system directory by default. Installation of Accuracy Checker is tested only via [virtual environment](https://docs.python.org/3/tutorial/venv.html).

Install the virtual environment:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv -p `which python3` <directory_for_environment>
```

Activate the virtual environment:

```bash
source <directory_for_environment>/bin/activate
```

Virtual environment can be deactivated using the following command:

```bash
deactivate
```

#### 2. Frameworks

The next step is installing backend frameworks for Accuracy Checker.

To evaluate some models, you need to install the required frameworks. Accuracy Checker supports the following frameworks:

- [OpenVINO](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started).
- [Caffe](https://caffe.berkeleyvision.org/installation.html).
- [MXNet](https://mxnet.apache.org/).
- [OpenCV DNN](https://docs.opencv.org/4.1.0/d2/de6/tutorial_py_setup_in_ubuntu.html).
- [TensorFlow](https://www.tensorflow.org/).
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/master/README.md).
- [PyTorch](https://pytorch.org/)
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)

You can use any of them or several at a time. For correct work, Accuracy Checker requires at least one. You can postpone installation of other frameworks and install them when they will be necessary.

### Install Accuracy Checker

If all prerequisites are installed, you are ready to install **Accuracy Checker**:

```bash
python3 setup.py install
```

Accuracy Checker is a modular tool and have some task-specific dependencies, all specific required modules can be found in `requirements.in` file.
 Instead of the standard installation, you can install only the core part of the tool without additional dependencies and manage them by yourself using the following command:

```bash
python setup.py install_core
```

#### Installation Troubleshooting

When previous version of the tool is already installed in the environment, in some cases, it can broke the new installation.
If you get a directory/file not found error, try manually removing the previous tool version from your environment or install the tool using following command in Accuracy Checker directory instead of setup.py install:

```bash
pip install --upgrade --force-reinstall .
```

#### Running the Tool inside IDE for Development Purposes

Accuracy Checker tool has an entry point for running in CLI, however, the majority of popular code editors or integrated development environments (IDEs) expect scripts as the starting point of application.
Sometimes it can be useful to run the tool as a script for debugging or enabling new models.
To use Accuracy Checker inside the IDE, you need to create a script in accuracy_checker root directory, for example, `<open_model_zoo>/tools/accuracy_checker/main.py`, with the following code:

```python
from accuracy_checker.main import main

if __name__ == '__main__':
    main()

```
Now, you can use this script for running Accuracy Checker in IDE.

#### Usage

You may test your installation and get familiar with Accuracy Checker by running a [sample](sample/README.md).

Each Open Model Zoo model can be evaluated using a configuration file. To learn more, refer to [How to use predefined configuration files](configs/README.md) guide.

Once you installed accuracy checker, you can evaluate your configurations using:

```bash
accuracy_check -c path/to/configuration_file -m /path/to/models -s /path/to/source/data -a /path/to/annotation
```

Use `-h, --help` to get the full list of command-line options. Some arguments are described below:

- `-c, --config` path to configuration file.
- `-m, --models` specifies directory in which models and weights declared in config file will be searched. You can also specify space-separated list of directories if you want to run the same configuration several times with models located in different directories or if you have the pipeline with several models.
- `-s, --source` specifies directory in which input images will be searched.
- `-a, --annotations` specifies directory in which annotation and meta files will be searched.
- `-d, --definitions` path to the global configuration file.
- `-e, --extensions` directory with InferenceEngine extensions.
- `-b, --bitstreams` directory with bitstream (for Inference Engine with fpga plugin).
- `-C, --converted_models` directory to store Model Optimizer converted models (used for DLSDK launcher only).
- `-tf, --target_framework` framework for infer.
- `-td, --target_devices` devices for infer. You can specify several devices using space as a delimiter.
- `--async_mode` allows run the tool in async mode if launcher supports it.
- `--num_requests` number requests for async execution. Allows override provided in config info. Default is `AUTO`
- `--model_attributes` directory with additional models attributes.
- `--subsample_size` dataset subsample size.
- `--shuffle` allows shuffle annotation during creation a subset if subsample_size argument is provided. Default is `True`.
- `--intermediate_metrics_results` enables intermediate metrics results printing. Default is `False`
- `--metrics_interval` number of iterations for updated metrics result printing if `--intermediate_metrics_results` flag enabled. Default is 1000.

You are also able to replace some command-line arguments with the environment variables for path prefixing. Supported list of variables includes:
* `DEFINITIONS_FILE` - equivalent of `-d`, `-definitions`.
* `DATA_DIR` -  equivalent of `-s`, `--source`.
* `MODELS_DIR` - equivalent of `-m`, `--models`.
* `EXTENSIONS` - equivalent of `-e`, `--extensions`.
* `ANNOTATIONS_DIR` - equivalent of `-a`, `--annotations`.
* `BITSTREAMS_DIR` - equivalent of `-b`, `--bitstreams`.
* `MODEL_ATTRIBUTES_DIR` - equivalent of `--model_attributes`.

#### Configuration

There is a config file, which declares the validation process.
Every validated model has to have its entry in the `models` list
with distinct `name` and other properties described below.

There is also a definitions file, which declares global options shared across all models.
Config file has priority over definitions file.

Example:

```yaml
models:
- name: model_name
  launchers:
    - framework: caffe
      model:   bvlc_alexnet.prototxt
      weights: bvlc_alexnet.caffemodel
      adapter: classification
      batch: 128
  datasets:
    - name: dataset_name
```
Optionally you can use global configuration. It may be useful for avoiding duplication if you have several models which need to be run on the same dataset.
Example of global definitions file can be found at `<omz_dir>/data/dataset_definitions.yml`. Global definitions will be merged with evaluation config in the runtime by dataset name.
Parameters of global configuration can be overwritten by local config. For example, if in definitions specified resize with destination size 224 and in the local config used resize with size 227, the value in config 227 will be used as resize parameter.
You can use field `global_definitions` for specifying the path to global definitions directly in the model config or via command-line arguments (`-d`, `--definitions`).

### Launchers

Launcher is a description of how your model should be executed.
Each launcher configuration starts with setting `framework` name.
Currently *caffe*, *dlsdk*, *mxnet*, *tf*, *tf2*, *tf_lite*, *opencv*, *onnx_runtime*, *pytorch*, *paddlepaddle* supported.
Launcher description can have differences.

- [How to configure Caffe launcher](accuracy_checker/launcher/caffe_launcher_readme.md)
- [How to configure OpenVINO launcher](accuracy_checker/launcher/dlsdk_launcher_readme.md)
- [How to configure OpenCV launcher](accuracy_checker/launcher/opencv_launcher_readme.md)
- [How to configure MXNet Launcher](accuracy_checker/launcher/mxnet_launcher_readme.md)
- [How to configure TensorFlow Launcher](accuracy_checker/launcher/tf_launcher_readme.md)
- [How to configure TensorFlow Lite Launcher](accuracy_checker/launcher/tf_lite_launcher_readme.md)
- [How to configure TensorFlow 2.0 Launcher](accuracy_checker/launcher/tf2_launcher_readme.md)
- [How to configure ONNX Runtime Launcher](accuracy_checker/launcher/onnx_runtime_launcher_readme.md)
- [How to configure PyTorch Launcher](accuracy_checker/launcher/pytorch_launcher_readme.md)
- [How to configure PaddlePaddle Launcher](accuracy_checker/launcher/pdpd_launcher_readme.md)

### Datasets

Dataset entry describes the data on which model should be evaluated,
all required preprocessing and postprocessing/filtering steps,
and metrics that will be used for evaluation.

If your dataset data is a well-known competition problem (COCO, Pascal VOC, and others) and/or can be potentially reused for other models,
it is reasonable to declare it in some global configuration file (`<omz_dir>/data/dataset_definitions.yml`). This way in your local configuration file you can provide only
`name`, and all required steps will be picked from the global one. To pass the path to this global configuration use `--definition` argument of CLI.

If you want to evaluate models using prepared config files and well-known datasets, you need to organize folders with validation datasets in a certain way. Find more detailed information about dataset preparation in [Dataset Preparation Guide](../../data/datasets.md).

Each dataset must have:

- `name` - unique identifier of your model/topology.
- `data_source` - path to directory where input data is stored.
- `metrics` - list of metrics that should be computed.

And optionally:
- `preprocessing` - list of preprocessing steps applied to input data. If you want calculated metrics to match reported, you must reproduce preprocessing from canonical paper of your topology or ask topology author about required steps.
- `postprocessing` - list of postprocessing steps.
- `reader` - approach for data reading. Default reader is `opencv_imread`.
- `segmentation_masks_source` - path to directory where gt masks for semantic segmentation task stored.

Also it must contain data related to annotation.
You can convert annotation in-place using:
- `annotation_conversion` - parameters for annotation conversion


or use existing annotation file and dataset meta:
- `annotation` - path to annotation file, you must **convert annotation to representation of dataset problem first**, you may choose one of the converters from *annotation-converters* if there is already a converter for your dataset or write your own.
- `dataset_meta` - path to metadata file (generated by converter).
Find more detailed information about annotation conversion in [Annotation Conversion Guide](accuracy_checker/annotation_converters/README.md).

Example of dataset definition:

```yaml
- name: dataset_name
  annotation: annotation.pickle
  data_source: images_folder

  preprocessing:
    - type: resize
      dst_width: 256
      dst_height: 256

    - type: normalization
      mean: imagenet

    - type: crop
      dst_width: 227
      dst_height: 227

  metrics:
    - type: accuracy
```

### Preprocessing, Metrics, Postprocessing

Each entry of preprocessing, metrics, postprocessing must have a `type` field 
with other options specific to the type. If you do not provide any other option, it
will be picked from the *definitions* file.

You can use the following instructions:

- [How to convert annotations](accuracy_checker/annotation_converters/README.md)
- [How to use preprocessing](accuracy_checker/preprocessor/README.md)
- [How to use postprocessing](accuracy_checker/postprocessor/README.md)
- [How to use metrics](accuracy_checker/metrics/README.md)
- [How to use readers](accuracy_checker/data_readers/README.md)

You may optionally provide `reference` field for metric, if you want the calculated metric
tested against a specific value (reported in canonical paper).

Some metrics support providing vector results, for example, mAP is able to return average precision for each detection class. You can change view mode for metric results using `presenter` (for example, `print_vector`, `print_scalar`).

Example:

```yaml
metrics:
- type: accuracy
  top_k: 5
  reference: 86.43
  threshold: 0.005
```

### Testing New Models

Typical workflow for testing a new model includes:

1. Convert annotation of your dataset. Use one of the converters from annotation-converters, or write your own if there is no converter for your dataset. You can find detailed instruction on how to use converters in [Annotation Conversion Guide](accuracy_checker/annotation_converters/README.md).
2. Choose one of the *adapters* or write your own. Adapter converts raw output produced by the framework to high-level problem-specific representation (for example, *ClassificationPrediction*, *DetectionPrediction*, and others).
3. Reproduce preprocessing, metrics and postprocessing from canonical paper.
4. Create entry in config file and execute.

### Customizing Evaluation

Standard Accuracy Checker validation pipeline: Annotation Reading -> Data Reading -> Preprocessing -> Inference -> Postprocessing -> Metrics.
In some cases, this validation pipeline can be unsuitable, for example, when you have a sequence of models. You can customize validation pipeline using your own evaluator.
Find more details about custom evaluations in the [related section](accuracy_checker/evaluators/custom_evaluators/README.md).
