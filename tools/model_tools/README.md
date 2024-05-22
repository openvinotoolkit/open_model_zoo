# Model Downloader and other automation tools

Open Model Zoo automation tools contain scripts that automate certain model-related tasks
based on configuration files in the models' directories.

* Model Downloader: `omz_downloader` downloads model files from online sources
  and, if necessary, patches them to make them more usable with Model
  Optimizer;

* Model Converter: `omz_converter` converts the models that are not in the
  OpenVINO™ IR format into that format using Model Optimizer.

*  Model Information Dumper: `omz_info_dumper` prints information about the models
  in a stable machine-readable format.

* Datasets' Data Downloader: `omz_data_downloader` copies datasets' data from installed location.

Please use these tools instead of attempting to parse the configuration files
directly. Their format is undocumented and may change in incompatible ways in
future releases.

> **TIP**: You also can work with the Model Downloader inside the OpenVINO™ [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench).
> [DL Workbench](@ref workbench_docs_Workbench_DG_Introduction) is a platform built upon OpenVINO™ and provides a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare
> performance of deep learning models on various Intel® architecture
> configurations. In the DL Workbench, you can use most of OpenVINO™ toolkit components.
> <br>
> Proceed to an [easy installation from Docker](@ref workbench_docs_Workbench_DG_Run_Locally) to get started.

## Installation

Model Downloader and other automation tools can be installed as part of the OpenVINO™ Development Tools Python package or from source if you need the latest changes.
To install the tools from the package, go to the [OpenVINO™ Development Tools PyPI page] (https://pypi.org/project/openvino-dev/) and follow the instructions.

To install the tools from source:

1. Install Python (version 3.6 or higher), [setuptools](https://pypi.org/project/setuptools/):

2. Install [openvino-dev](https://pypi.org/project/openvino-dev/) Python\* package to obtain Model Optimizer and Post-Training Optimization Toolkit:

```sh
pip install openvino-dev
```
> **NOTE**: openvino-dev version should be the same as OMZ Tools. For example, if you are using OMZ Tools for 2021.4.2 then install openvino-dev==2021.4.2.

3. Install the `omz-tools` with the following command:

```sh
# Navigate to the open_model_zoo/tools/model_tools directory
pip install --upgrade pip
pip install .
```

> **NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`. You may also need to [install pip](https://pip.pypa.io/en/stable/installation/).
> For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-pip`.
> If you are using pip version lower than 21.3, you also need to set OMZ_ROOT variable: `export OMZ_ROOT=<omz_dir>`

To convert models from certain frameworks, you may also need to install
additional dependencies.

<!--
@sphinxdirective

.. tab:: PyTorch

  .. code-block:: sh

    python -mpip install --user -r ./requirements-pytorch.in

.. tab:: TensorFlow

  .. code-block:: sh

    python -mpip install --user -r ./requirements-tensorflow.in

@endsphinxdirective
-->

## Model Downloader Usage

The basic usage is to run the script like this:

```sh
omz_downloader --all
```

This will download all models. The `--all` option can be replaced with
other filter options to download only a subset of models. See the "Shared options"
section.

### Model Downloader Starting Parameters

<!--
@sphinxdirective

+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| Parameter                 | Explanation                                                                                                                                                                                                                                                                                                                                                                                      | Example                                                                             |
+===========================+==================================================================================================================================================================================================================================================================================================================================================================================================+=====================================================================================+
| ``-o``/``--output_dir``   | By default, the script will download models into a directory tree rooted in the current directory. Use this parameter to download into a different directory.                                                                                                                                                                                                                                    | ``omz_downloader --all --output_dir my/download/directory``                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| ``--precisions``          | Specify comma separated precisions of weights to be downloaded                                                                                                                                                                                                                                                                                                                                   | ``omz_downloader --name face-detection-retail-0004 --precisions FP16,FP16-INT8``   |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| ``--num_attempts``        | By default, the script will attempt to download each file only once. Use this parameter to change that and increase the robustness of the download process                                                                                                                                                                                                                                       | ``omz_downloader --all --num_attempts 5 # attempt each download five times``       |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| ``--cache_dir``           | Make the script use the specified directory as a cache. The script will place a copy of each downloaded file in the cache, or, if it is already there, retrieve it from the cache instead of downloading it again. The cache format is intended to remain compatible in future Open Model Zoo versions, so you can use a cache to avoid redownloading most files when updating Open Model Zoo.   | ``omz_downloader --all --cache_dir my/cache/directory``                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| ``-j``/``--jobs``         | The script downloads files for multiple models concurrently.                                                                                                                                                                                                                                                                                                                                     | ``omz_downloader --all -j8 # download up to 8 models at a time``                   |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| ``--progress_format``     | By default, the script outputs progress information as unstructured, human-readable text. You can also set this option to `text` to explicitly request the default text format. When this option is set to `json`, the script's standard output is replaced by a machine-readable progress report, whose format is documented in the "JSON progress report format" section. This option does not affect errors and warnings, which will still be printed to the standard error stream in a human-readable format. Use this option, if you want to consume progress information programmatically.                                                                                                                                                                                                                         | ``omz_downloader --all --progress_format=json``                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
@endsphinxdirective
-->

See the "Shared options" section for information on other options accepted by
the script.

### JSON Progress Report Format

This section documents the format of the progress report produced by the script
when the `--progress_format=json` option is specified.

The report consists of a sequence of events, where each event is represented
by a line containing a JSON-encoded object. Each event has a member with the
name `$type` whose value determines the type of the event, as well as which
additional members it contains.

The following event types are currently defined:

<!--
@sphinxdirective

+------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Event type                         | Additional members                                                      | Explanation                                                                                                                                                                                                                                                                                                                                    |
+====================================+=========================================================================+================================================================================================================================================================================================================================================================================================================================================+
| ``model_download_begin``           | ``model`` (string), ``num_files`` (integer)                             | The script started downloading the model named by ``model``. ``num_files`` is the number of files that will be downloaded for this model. This event will always be followed by a corresponding ``model_download_end`` event.                                                                                                                  |
+------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model_download_end``             | ``model`` (string), ``successful`` (boolean)                            | The script stopped downloading the model named by ``model``. ``successful`` is true if every file was downloaded successfully.                                                                                                                                                                                                                 |
+------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model_file_download_begin``      | ``model`` (string), ``model_file`` (string), ``size`` (integer)         | The script started downloading the file named by ``model_file`` of the model named by ``model``. ``size`` is the size of the file in bytes. This event will always occur between ``model_download_begin`` and ``model_download_end`` events for the model, and will always be followed by a corresponding ``model_file_download_end`` event.   |
+------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model_file_download_end``        | ``model`` (string), ``model_file`` (string), ``successful`` (boolean)   | The script stopped downloading the file named by ``model_file`` of the model named by ``model``. ``successful`` is true if the file was downloaded successfully.                                                                                                                                                                               |
+------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model_file_download_progress``   | ``model`` (string), ``model_file`` (string), ``size`` (integer)         | The script downloaded ``size`` bytes of the file named by ``model_file`` of the model named by ``model`` so far. Note that ``size`` can decrease in a subsequent event if the download is interrupted and retried. This event will always occur between ``model_file_download_begin`` and ``model_file_download_end`` events for the file.     |
+------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model_postprocessing_begin``     | ``model``                                                               | The script started post-download processing on the model named by ``model``. This event will always be followed by a corresponding ``model_postprocessing_end`` event.                                                                                                                                                                         |
+------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model_postprocessing_end``       | ``model``                                                               | The script stopped post-download processing on the model named by ``model``.                                                                                                                                                                                                                                                                   |
+------------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
@endsphinxdirective
-->

Additional event types and members may be added in the future.

Tools parsing the machine-readable format should avoid relying on undocumented details.
In particular:

* Tools should not assume that any given event will occur for a given model/file
  (unless specified otherwise above) or will only occur once.

* Tools should not assume that events will occur in a certain order beyond
  the ordering constraints specified above. In particular, when the `--jobs` option
  is set to a value greater than 1, event sequences for different files or models
  may get interleaved.

## Model Converter Usage

The basic usage is to run the script like this:

```sh
omz_converter --all
```

This will convert all models into the OpenVINO™ IR format. Models that
were originally in that format are ignored. Models in PyTorch format will be
converted in ONNX format first.

The `--all` option can be replaced with other filter options to convert only
a subset of models. See the "Shared options" section.

### Model Converter Starting Parameters

<!--
@sphinxdirective

+-----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Parameter                   | Explanation                                                                                                                                                                                                                                                      | Example                                                                                          |
+=============================+==================================================================================================================================================================================================================================================================+==================================================================================================+
| ``-d``/``--download_dir``   | The current directory must be the root of a download tree created by the model downloader. Use this parameter to specify a different download tree path.                                                                                                         | ``omz_converter --all --download_dir my/download/directory``                                    |
+-----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| ``-o``/``--output_dir``     | By default, the script will download models into a directory tree rooted in the current directory. Use this parameter to download into a different directory. Note: models in intermediate format are placed to this directory too.                              | ``omz_converter --all --output_dir my/output/directory``                                        |
+-----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| ``--precisions``            | By default, the script will produce models in every precision that is supported for conversion. Use this parameter to only produce models in a specific precision. If the specified precision is not supported for a model, that model will be skipped.          | ``omz_converter --all --precisions=FP16``                                                       |
+-----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| ``--add_mo_arg``            | Add extra Model Optimizer arguments to the ones specified in the model configuration. The option can be repeated to add multiple arguments                                                                                                                       | ``omz_converter --name=densenet-121-tf --add_mo_arg=--reverse_input_channels --add_mo_arg=--silent``   |
+-----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| ``-j``/``--jobs``           | Run multiple conversion commands concurrently. The argument to the option must be either a maximum number of concurrently executed commands, or "auto", in which case the number of CPUs in the system is used. By default, all commands are run sequentially.   | ``omz_converter --all -j8 # run up to 8 commands at a time``                                    |
+-----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| ``--dry_run``               | Print the conversion commands without actually running them..                                                                                                                                                                                                    | ``omz_converter --all --dry_run``                                                               |
+-----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| ``-p``/``--python``         | By default, the script will run Model Optimizer using the same Python executable that was used to run the script itself. Apply this parameter to use a different Python executable.                                                                              | ``omz_converter --all --python my/python``                                                      |
+-----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
@endsphinxdirective
-->

The script will attempt to locate Model Optimizer using several methods:

1. If the `--mo` option was specified, then its value will be used as the path
   to the script to run:

   ```sh
   omz_converter --all --mo my/openvino/path/model_optimizer/mo.py
   ```

2. Otherwise, if the selected Python executable can find the `mo` entrypoint,
   then it will be used.

3. Otherwise, if the OpenVINO&trade; toolkit's `setupvars.sh`/`setupvars.bat`
   script has been executed, the environment variables set by that script will
   be used to locate Model Optimizer within the toolkit.

4. Otherwise, the script will fail.

## Model Information Dumper Usage

The basic usage is to run the script like this:

```sh
omz_info_dumper --all
```

The other options accepted by the script are described in the "Shared options" section.

This will print to standard output information about all models. The script's output is a JSON array, each element of which is a JSON object
describing a single model. Each such object has the following keys:


<!--
@sphinxdirective

+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                            | Explanation                                                                                                                                                                                                                                                                         |
+======================================+=====================================================================================================================================================================================================================================================================================+
| ``name``                             | The identifier of the model, as accepted by the ``--name`` option.                                                                                                                                                                                                                  |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``composite_model_name``             | The identifier of the composite model name, if the model is a part of composition of several models (e.g. encoder-decoder), otherwise - ``null``                                                                                                                                    |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``description``                      | Text describing the model. Paragraphs are separated by line feed characters.                                                                                                                                                                                                        |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``framework``                        | A string identifying the framework whose format the model is downloaded in. Current possible values are ``dldt`` (Inference Engine IR), ``caffe``, ``onnx``, ``pytorch`` and ``tf`` (TensorFlow). Additional possible values might be added in the future.               |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``license_url``                      | A URL for the license that the model is distributed under.                                                                                                                                                                                                                          |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``accuracy_config``                      | Path to the model accuracy config in the user's file system.                                                                                                                                                                                                                          |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model_config``                      | Path to the model configuration file in the user's file system.                                                                                                                                                                                                                          |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``precisions``                       | The list of precisions that the model has IR files for. For models downloaded in a format other than the Inference Engine IR format, these are the precisions that the model converter can produce IR files in. Current possible values are:                                        |
|                                      | * `FP16`                                                                                                                                                                                                                                                                            |
|                                      | * `FP16-INT1`                                                                                                                                                                                                                                                                       |
|                                      | * `FP16-INT8`                                                                                                                                                                                                                                                                       |
|                                      | * `FP32`                                                                                                                                                                                                                                                                            |
|                                      | * `FP32-INT1`                                                                                                                                                                                                                                                                       |
|                                      | * `FP32-INT8`                                                                                                                                                                                                                                                                       |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``subdirectory``                     | The subdirectory of the output tree into which the downloaded or converted files will be placed by the downloader or the converter, respectively.                                                                                                                                   |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``task_type``                        | a string identifying the type of task that the model performs. Current possible values are:                                                                                                                                                                                         |
|                                      |                                                                                                                                                                                                                                                                                     |
|                                      | * `action_recognition`                                                                                                                                                                                                                                                              |
|                                      | * `classification`                                                                                                                                                                                                                                                                  |
|                                      | * `colorization`                                                                                                                                                                                                                                                                    |
|                                      | * `detection`                                                                                                                                                                                                                                                                       |
|                                      | * `face_recognition`                                                                                                                                                                                                                                                                |
|                                      | * `feature_extraction`                                                                                                                                                                                                                                                              |
|                                      | * `head_pose_estimation`                                                                                                                                                                                                                                                            |
|                                      | * `human_pose_estimation`                                                                                                                                                                                                                                                           |
|                                      | * `image_inpainting`                                                                                                                                                                                                                                                                |
|                                      | * `image_processing`                                                                                                                                                                                                                                                                |
|                                      | * `image_translation`                                                                                                                                                                                                                                                               |
|                                      | * `instance_segmentation`                                                                                                                                                                                                                                                           |
|                                      | * `machine_translation`                                                                                                                                                                                                                                                             |
|                                      | * `monocular_depth_estimation`                                                                                                                                                                                                                                                      |
|                                      | * `named_entity_recognition`                                                                                                                                                                                                                                                        |
|                                      | * `noise_suppression`                                                                                                                                                                                                                                                               |
|                                      | * `object_attributes`                                                                                                                                                                                                                                                               |
|                                      | * `optical_character_recognition`                                                                                                                                                                                                                                                   |
|                                      | * `place_recognition`                                                                                                                                                                                                                                                               |
|                                      | * `question_answering`                                                                                                                                                                                                                                                              |
|                                      | * `salient_object_detection`                                                                                                                                                                                                                                                        |
|                                      | * `semantic_segmentation`                                                                                                                                                                                                                                                           |
|                                      | * `sound_classification`                                                                                                                                                                                                                                                            |
|                                      | * `speech_recognition`                                                                                                                                                                                                                                                              |
|                                      | * `style_transfer`                                                                                                                                                                                                                                                                  |
|                                      | * `text_prediction`                                                                                                                                                                                                                                                                 |
|                                      | * `text_to_speech`                                                                                                                                                                                                                                                                  |
|                                      | * `time_series`                                                                                                                                                                                                                                                                     |
|                                      | * `token_recognition`                                                                                                                                                                                                                                                               |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``input_info``                       | The list of inputs containing the information about input name, shape and layout.                                                                                                                                                                                                   |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model_stages``                     | The list of model stages, in case if the model is a composition of several models,  otherwise the list is empty                                                                                                                                                                     |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
@endsphinxdirective
-->

## Shared Options

The are certain options that all tools accept.

`-h`/`--help` can be used to print a help message:

```sh
omz_TOOL --help
```
There are several mutually exclusive filter options that select the models the
tool will process:

<!--
@sphinxdirective

+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
| Parameter    | Explanation                                                                                                                                                                                                                                                                       | Example                                   |
+==============+===================================================================================================================================================================================================================================================================================+===========================================+
| ``--all``    | Selects all models                                                                                                                                                                                                                                                                | ``omz_TOOL --all``                        |
+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
| ``--name``   | takes a comma-separated list of patterns and selects models that match at least one of these patterns. The patterns may contain shell-style wildcards. See https://docs.python.org/3/library/fnmatch.html for a full description of the pattern syntax. For composite models, the name of composite model is accepted, as well as the names of individual models it consists of.   | ``omz_TOOL --name 'wavernn,densenet-*'``    |
+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
| ``--list``   | takes a path to a file that must contain a list of patterns and selects models that match at least one of those patterns. For composite models, the name of composite model is accepted, as well as the names of individual models it consists of                                 | ``omz_TOOL --list my.lst``                |
+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
@endsphinxdirective
-->


To see the available models, you can use the `--print_all` option. When this
option is specified, the tool will print all model names defined in the
configuration file and exit:

```
$ omz_TOOL --print_all
action-recognition-0001-decoder
action-recognition-0001-encoder
age-gender-recognition-retail-0013
driver-action-recognition-adas-0002-decoder
driver-action-recognition-adas-0002-encoder
emotions-recognition-retail-0003
face-detection-adas-0001
face-detection-retail-0004
face-detection-retail-0005
[...]
```

Either `--print_all` or one of the filter options must be specified.


## Datasets' Data Downloader Usage

The usage is to run the script like this:

```sh
omz_data_downloader -o my/output/directory
```

This will copy datasets' data from installed location to the specified location.
If `-o`/`--output_dir` option is not set, the files will be copied into a directory tree rooted in the current directory.


__________

OpenVINO is a trademark of Intel Corporation or its subsidiaries in the U.S.
and/or other countries.


Copyright &copy; 2018-2024 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
