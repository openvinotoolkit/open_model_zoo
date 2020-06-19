# Deep Learning accuracy validation framework

## Installation

### Prerequisites

Install prerequisites first:

#### 1. Python

**accuracy checker** uses **Python 3**. Install it first:

- [Python3](https://www.python.org/downloads/), [setuptools](https://pypi.org/project/setuptools/):

```bash
sudo apt-get install python3 python3-dev python3-setuptools python3-pip
```

Python setuptools and python package manager (pip) install packages into system directory by default. Installation of accuracy checker tested only via [virtual environment](https://docs.python.org/3/tutorial/venv.html).

In order to use virtual environment you should install it first:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv -p `which python3` <directory_for_environment>
```

Before starting to work inside virtual environment, it should be activated:

```bash
source <directory_for_environment>/bin/activate
```

Virtual environment can be deactivated using command

```bash
deactivate
```

#### 2. Frameworks

The next step is installing backend frameworks for Accuracy Checker.

In order to evaluate some models required frameworks have to be installed. Accuracy-Checker supports these frameworks:

- [OpenVINO](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started).
- [Caffe](accuracy_checker/launcher/caffe_installation_readme.md).
- [MXNet](https://mxnet.apache.org/).
- [OpenCV DNN](https://docs.opencv.org/4.1.0/d2/de6/tutorial_py_setup_in_ubuntu.html).
- [TensorFlow](https://www.tensorflow.org/).
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/master/README.md).
- [PyTorch](https://pytorch.org/)

You can use any of them or several at a time. For correct work, Accuracy Checker requires at least one. You are able postpone installation of other frameworks and install them when they will be necessary.

### Install accuracy checker

If all prerequisite are installed, then you are ready to install **accuracy checker**:

```bash
python3 setup.py install
```

Accuracy Checker is modular tool and have some task-specific dependencies, all specific required modules can be found in `requirements.in` file.
You can install only core part of the tool without additional dependencies and manage them by your-self using following command instead of standard installation:

```bash
python setup.py install_core
```

#### Usage

You may test your installation and get familiar with accuracy checker by running [sample](sample/README.md).

Once you installed accuracy checker you can evaluate your configurations with:

```bash
accuracy_check -c path/to/configuration_file -m /path/to/models -s /path/to/source/data -a /path/to/annotation
```

All relative paths in config files will be prefixed with values specified in command line:

- `-c, --config` path to configuration file.
- `-m, --models` specifies directory in which models and weights declared in config file will be searched. You also can specify space separated list of directories if you want to run the same configuration several times with models located in different directories or if you have the pipeline with several models.
- `-s, --source` specifies directory in which input images will be searched.
- `-a, --annotations` specifies directory in which annotation and meta files will be searched.

You may refer to `-h, --help` to full list of command line options. Some optional arguments are:

- `-d, --definitions` path to the global configuration file.
- `-e, --extensions` directory with InferenceEngine extensions.
- `-b, --bitstreams` directory with bitstream (for Inference Engine with fpga plugin).
- `-C, '--converted_models` directory to store Model Optimizer converted models (used for DLSDK launcher only).
- `-tf, --target_framework` framework for infer.
- `-td, --target_devices` devices for infer. You can specify several devices using space as a delimiter.
- `--async_mode` allows run the tool in async mode if launcher support it.
- `--num_requests` number requests for async execution. Allows override provided in config info. Default is `AUTO`
- `--model_attributes` directory with additional models attributes.
- `--subsample_size` dataset subsample size.
- `--shuffle` allow shuffle annotation during creation a subset if subsample_size argument is provided. Default is `True`.

You are also able to replace some command line arguments with environment variables for path prefixing. Supported following list of variables:
* `DATA_DIR` -  equivalent of `-s`, `--source`.
* `MODELS_DIR` - equivalent of `-m`, `--models`.
* `EXTENSIONS` - equivalent of `-e`, `--extensions`.
* `ANNOTATIONS_DIR` - equivalent of `-a`, `--annotations`.
* `BITSTREAMS_DIR` - equivalent of `-b`, `--bitstreams`.
* `MODEL_ATTRIBUTES_DIR` - equivalent of `--model_attributes`.

#### Configuration

There is config file which declares validation process.
Every validated model has to have its entry in `models` list
with distinct `name` and other properties described below.

There is also definitions file, which declares global options shared across all models.
Config file has priority over definitions file.

example:

```yaml
models:
- name: model_name
  launchers:
    - framework: caffe
      model:   public/alexnet/caffe/bvlc_alexnet.prototxt
      weights: public/alexnet/caffe/bvlc_alexnet.caffemodel
      adapter: classification
      batch: 128
  datasets:
    - name: dataset_name
```
Optionally you can use global configuration. It can be useful for avoiding duplication if you have several models which should be run on the same dataset.
Example of global definitions file can be found [here](dataset_definitions.yml). Global definitions will be merged with evaluation config in the runtime by dataset name.
Parameters of global configuration can be overwritten by local config (e.g. if in definitions specified resize with destination size 224 and in the local config used resize with size 227, the value in config - 227 will be used as resize parameter)
You can use field `global_definitions` for specifying path to global definitions directly in the model config or via command line arguments (`-d`, `--definitions`).

### Launchers

Launcher is a description of how your model should be executed.
Each launcher configuration starts with setting `framework` name. Currently *caffe*, *dlsdk*, *mxnet*, *tf*, *tf_lite*, *opencv*, *onnx_runtime* supported. Launcher description can have differences.
Please view:

- [how to configure Caffe launcher](accuracy_checker/launcher/caffe_launcher_readme.md).
- [how to configure DLSDK launcher](accuracy_checker/launcher/dlsdk_launcher_readme.md).
- [how to configure OpenCV launcher](accuracy_checker/launcher/opencv_launcher_readme.md).
- [how to configure MXNet Launcher](accuracy_checker/launcher/mxnet_launcher_readme.md).
- [how to configure TensorFlow Launcher](accuracy_checker/launcher/tf_launcher_readme.md).
- [how to configure TensorFlow Lite Launcher](accuracy_checker/launcher/tf_lite_launcher_readme.md).
- [how to configure ONNX Runtime Launcher](accuracy_checker/launcher/onnx_runtime_launcher_readme.md).
- [how to configure PyTorch Launcher](accuracy_checker/launcher/pytorch_launcher_readme.md)

### Datasets

Dataset entry describes data on which model should be evaluated,
all required preprocessing and postprocessing/filtering steps,
and metrics that will be used for evaluation.

If your dataset data is a well-known competition problem (COCO, Pascal VOC, ...) and/or can be potentially reused for other models
it is reasonable to declare it in some global configuration file (*definition* file). This way in your local configuration file you can provide only
`name` and all required steps will be picked from global one. To pass path to this global configuration use `--definition` argument of CLI.

If you want to evaluate models using prepared config files and well-known datasets, you need to organize folders with validation datasets in a certain way. More detailed information about dataset preparation you can find in [Dataset Preparation Guide](../../datasets.md).

Each dataset must have:

- `name` - unique identifier of your model/topology.
- `data_source`: path to directory where input data is stored.
- `metrics`: list of metrics that should be computed.

And optionally:
- `preprocessing`: list of preprocessing steps applied to input data. If you want calculated metrics to match reported, you must reproduce preprocessing from canonical paper of your topology or ask topology author about required steps.
- `postprocessing`: list of postprocessing steps.
- `reader`: approach for data reading. Default reader is `opencv_imread`.
- `segmentation_masks_source` - path to directory where gt masks for semantic segmentation task stored.

Also it must contain data related to annotation.
You can convert annotation inplace using:
- `annotation_conversion`: parameters for annotation conversion


or use existing annotation file and dataset meta:
- `annotation` - path to annotation file, you must **convert annotation to representation of dataset problem first**, you may choose one of the converters from *annotation-converters* if there is already converter for your dataset or write your own.
- `dataset_meta`: path to metadata file (generated by converter).
More detailed information about annotation conversion you can find in [Annotation Conversion Guide](accuracy_checker/annotation_converters/README.md).

example of dataset definition:

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

Each entry of preprocessing, metrics, postprocessing must have `type` field,
other options are specific to type. If you do not provide any other option, then it
will be picked from *definitions* file.

You can find useful following instructions:

- [how to convert annotations](accuracy_checker/annotation_converters/README.md)
- [how to use preprocessings](accuracy_checker/preprocessor/README.md).
- [how to use postprocessings](accuracy_checker/postprocessor/README.md).
- [how to use metrics](accuracy_checker/metrics/README.md).
- [how to use readers](accuracy_checker/data_readers/README.md).

You may optionally provide `reference` field for metric, if you want calculated metric
tested against specific value (i.e. reported in canonical paper).

Some metrics support providing vector results ( e. g. mAP is able to return average precision for each detection class). You can change view mode for metric results using `presenter` (e.g. `print_vector`, `print_scalar`).

example:

```yaml
metrics:
- type: accuracy
  top_k: 5
  reference: 86.43
  threshold: 0.005
```

### Testing new models

Typical workflow for testing new model include:

1. Convert annotation of your dataset. Use one of the converters from annotation-converters, or write your own if there is no converter for your dataset. You can find detailed instruction how to use converters in [Annotation Conversion Guide](accuracy_checker/annotation_converters/README.md).
2. Choose one of *adapters* or write your own. Adapter converts raw output produced by framework to high level problem specific representation (e.g. *ClassificationPrediction*, *DetectionPrediction*, etc).
3. Reproduce preprocessing, metrics and postprocessing from canonical paper.
4. Create entry in config file and execute.

### Customizing Evaluation

Standard Accuracy Checker validation pipeline: Annotation Reading -> Data Reading -> Preprocessing -> Inference -> Postprocessing -> Metrics.
In some cases it can be unsuitable (e.g. if you have sequence of models). You are able to customize validation pipeline using own evaluator.
More details about custom evaluations can be found in [related section](accuracy_checker/evaluators/custom_evaluators/README.md).
