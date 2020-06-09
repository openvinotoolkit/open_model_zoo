Model Downloader and other automation tools
===========================================

This directory contains scripts that automate certain model-related tasks
based on configuration files in the models' directories.

* `downloader.py` (model downloader) downloads model files from online sources
  and, if necessary, patches them to make them more usable with Model
  Optimizer;

* `converter.py` (model converter) converts the models that are not in the
  Inference Engine IR format into that format using Model Optimizer.

* `quantizer.py` (model quantizer) quantizes full-precision models in the IR
  format into low-precision versions using Post-Training Optimization Toolkit.

* `info_dumper.py` (model information dumper) prints information about the models
  in a stable machine-readable format.

Please use these tools instead of attempting to parse the configuration files
directly. Their format is undocumented and may change in incompatible ways in
future releases.

Prerequisites
-------------

1. Install Python (version 3.5.2 or higher)
2. Install the tools' dependencies with the following command:

```sh
python3 -mpip install --user -r ./requirements.in
```

For the model converter, you will also need to install the OpenVINO&trade;
toolkit and the prerequisite libraries for Model Optimizer. See the
[OpenVINO toolkit documentation](https://docs.openvinotoolkit.org/) for details.

If you using models from PyTorch or Caffe2 framework, you will also need to use intermediate
conversion to ONNX format. To use automatic conversion install additional dependencies.

For models from PyTorch:
```sh
python3 -mpip install --user -r ./requirements-pytorch.in
```
For models from Caffe2:
```sh
python3 -mpip install --user -r ./requirements-caffe2.in
```

When running the model downloader with Python 3.5.x on macOS, you may encounter
an error similar to the following:

> requests.exceptions.SSLError: [...] (Caused by SSLError(SSLError(1, '[SSL: TLSV1_ALERT_PROTOCOL_VERSION]
tlsv1 alert protocol version (\_ssl.c:719)'),))

You can work around this by installing additional packages:

```sh
python3 -mpip install --user 'requests[security]'
```

Alternatively, upgrade to Python 3.6 or a later version.

Model downloader usage
----------------------

The basic usage is to run the script like this:

```sh
./downloader.py --all
```

This will download all models. The `--all` option can be replaced with
other filter options to download only a subset of models. See the "Shared options"
section.

By default, the script will download models into a directory tree rooted
in the current directory. To download into a different directory, use
the `-o`/`--output_dir` option:

```sh
./downloader.py --all --output_dir my/download/directory
```

You may use `--precisions` flag to specify comma separated precisions of weights
to be downloaded.

```sh
./downloader.py --name face-detection-retail-0004 --precisions FP16,INT8
```

By default, the script will attempt to download each file only once. You can use
the `--num_attempts` option to change that and increase the robustness of the
download process:

```sh
./downloader.py --all --num_attempts 5 # attempt each download five times
```

You can use the `--cache_dir` option to make the script use the specified directory
as a cache. The script will place a copy of each downloaded file in the cache, or,
if it is already there, retrieve it from the cache instead of downloading it again.

```sh
./downloader.py --all --cache_dir my/cache/directory
```

The cache format is intended to remain compatible in future Open Model Zoo
versions, so you can use a cache to avoid redownloading most files when updating
Open Model Zoo.

By default, the script outputs progress information as unstructured, human-readable
text. If you want to consume progress information programmatically, use the
`--progress_format` option:

```sh
./downloader.py --all --progress_format=json
```

When this option is set to `json`, the script's standard output is replaced by
a machine-readable progress report, whose format is documented in the
"JSON progress report format" section. This option does not affect errors and
warnings, which will still be printed to the standard error stream in a
human-readable format.

You can also set this option to `text` to explicitly request the default text
format.

The script can download files for multiple models concurrently. To enable this,
use the `-j`/`--jobs` option:

```sh
./downloader.py --all -j8 # download up to 8 models at a time
```

See the "Shared options" section for information on other options accepted by
the script.

### JSON progress report format

This section documents the format of the progress report produced by the script
when the `--progress_format=json` option is specified.

The report consists of a sequence of events, where each event is represented
by a line containing a JSON-encoded object. Each event has a member with the
name `$type` whose value determines the type of the event, as well as which
additional members it contains.

The following event types are currently defined:

* `model_download_begin`

  Additional members: `model` (string), `num_files` (integer).

  The script started downloading the model named by `model`.
  `num_files` is the number of files that will be downloaded for this model.

  This event will always be followed by a corresponding `model_download_end` event.

* `model_download_end`

  Additional members: `model` (string), `successful` (boolean).

  The script stopped downloading the model named by `model`.
  `successful` is true if every file was downloaded successfully.

* `model_file_download_begin`

  Additional members: `model` (string), `model_file` (string), `size` (integer).

  The script started downloading the file named by `model_file` of the model named
  by `model`. `size` is the size of the file in bytes.

  This event will always occur between `model_download_begin` and `model_download_end`
  events for the model, and will always be followed by a corresponding
  `model_file_download_end` event.

* `model_file_download_end`

  Additional members: `model` (string), `model_file` (string), `successful` (boolean).

  The script stopped downloading the file named by `model_file` of the model named
  by `model`. `successful` is true if the file was downloaded successfully.

* `model_file_download_progress`

  Additional members: `model` (string), `model_file` (string), `size` (integer).

  The script downloaded `size` bytes of the file named by `model_file` of
  the model named by `model` so far. Note that `size` can decrease in a subsequent
  event if the download is interrupted and retried.

  This event will always occur between `model_file_download_begin` and
  `model_file_download_end` events for the file.

* `model_postprocessing_begin`

  Additional members: `model`.

  The script started post-download processing on the model named by `model`.

  This event will always be followed by a corresponding `model_postprocessing_end` event.

* `model_postprocessing_end`

  Additional members: `model`.

  The script stopped post-download processing on the model named by `model`.

Additional event types and members may be added in the future.

Tools parsing the machine-readable format should avoid relying on undocumented details.
In particular:

* Tools should not assume that any given event will occur for a given model/file
  (unless specified otherwise above) or will only occur once.

* Tools should not assume that events will occur in a certain order beyond
  the ordering constraints specified above. In particular, when the `--jobs` option
  is set to a value greater than 1, event sequences for different files or models
  may get interleaved.

Model converter usage
---------------------

The basic usage is to run the script like this:

```sh
./converter.py --all
```

This will convert all models into the Inference Engine IR format. Models that
were originally in that format are ignored. Models in PyTorch and Caffe2 formats will be
converted in ONNX format first.

The `--all` option can be replaced with other filter options to convert only
a subset of models. See the "Shared options" section.

The current directory must be the root of a download tree created by the model
downloader. To specify a different download tree path, use the `-d`/`--download_dir`
option:

```sh
./converter.py --all --download_dir my/download/directory
```

By default, the converted models are placed into the download tree. To place them
into a different directory tree, use the `-o`/`--output_dir` option:

```sh
./converter.py --all --output_dir my/output/directory
```
>Note: models in intermediate format are placed to this directory too.

By default, the script will produce models in every precision that is supported
for conversion. To only produce models in a specific precision, use the `--precisions`
option:

```sh
./converter.py --all --precisions=FP16
```

If the specified precision is not supported for a model, that model will be skipped.

The script will attempt to locate Model Optimizer using the environment
variables set by the OpenVINO&trade; toolkit's `setupvars.sh`/`setupvars.bat`
script. You can override this heuristic with the `--mo` option:

```sh
./converter.py --all --mo my/openvino/path/model_optimizer/mo.py
```

You can add extra Model Optimizer arguments to the ones specified in the model
configuration by using the `--add_mo_arg` option. The option can be repeated
to add multiple arguments:

```sh
./converter.py --name=caffenet --add_mo_arg=--reverse_input_channels --add_mo_arg=--silent
```

By default, the script will run Model Optimizer using the same Python executable
that was used to run the script itself. To use a different Python executable,
use the `-p`/`--python` option:

```sh
./converter.py --all --python my/python
```

The script can run multiple conversion commands concurrently. To enable this,
use the `-j`/`--jobs` option:

```sh
./converter.py --all -j8 # run up to 8 commands at a time
```

The argument to the option must be either a maximum number of concurrently
executed commands, or "auto", in which case the number of CPUs in the system is used.
By default, all commands are run sequentially.

The script can print the conversion commands without actually running them.
To do this, use the `--dry_run` option:

```sh
./converter.py --all --dry_run
```

See the "Shared options" section for information on other options accepted by
the script.

Model quantizer usage
---------------------

Before you run the model quantizer, you must prepare a directory with
the datasets required for the quantization process. This directory will be
referred to as `<DATASET_DIR>` below. You can find more detailed information
about dataset preparation in the [Dataset Preparation Guide](../../datasets.md).

The basic usage is to run the script like this:

```sh
./quantizer.py --all --dataset_dir <DATASET_DIR>
```

This will quantize all models for which quantization is supported. Other models
are ignored.

The `--all` option can be replaced with other filter options to quantize only
a subset of models. See the "Shared options" section.

The current directory must be the root of a tree of model files create by the model
converter. To specify a different model tree path, use the `--model_dir` option:

```sh
./quantizer.py --all --dataset_dir <DATASET_DIR> --model_dir my/model/directory
```

By default, the quantized models are placed into the same model tree. To place them
into a different directory tree, use the `-o`/`--output_dir` option:

```sh
./quantizer.py --all --dataset_dir <DATASET_DIR> --output_dir my/output/directory
```

By default, the script will produce models in every precision that is supported
as a quantization output. To only produce models in a specific precision, use
the `--precisions` option:

```sh
./quantizer.py --all --dataset_dir <DATASET_DIR> --precisions=FP16-INT8
```

The script will attempt to locate Post-Training Optimization Toolkit using
the environment variables set by the OpenVINO&trade; toolkit's `setupvars.sh`/`setupvars.bat`
script. You can override this heuristic with the `--pot` option:

```sh
./quantizer.py --all --dataset_dir <DATASET_DIR> --pot my/openvino/path/post_training_optimization_toolkit/main.py
```

By default, the script will run Post-Training Optimization Toolkit using the same
Python executable that was used to run the script itself. To use a different
Python executable, use the `-p`/`--python` option:

```sh
./quantizer.py --all --dataset_dir <DATASET_DIR> --python my/python
```

It's possible to specify a target device for Post-Training Optimization Toolkit
to optimize for, by using the `--target_device` option:

```sh
./quantizer.py --all --dataset_dir <DATASET_DIR> --target_device VPU
```

The supported values are those accepted by the "target_device" option in
Post-Training Optimization Toolkit's config files. If this option is unspecified,
Post-Training Optimization Toolkit's default is used.

The script can print the quantization commands without actually running them.
To do this, use the `--dry_run` option:

```sh
./quantizer.py --all --dataset_dir <DATASET_DIR> --dry_run
```

With this option specified, the configuration file for Post-Training Optimization
Toolkit will still be created, so that you can inspect it.

See the "Shared options" section for information on other options accepted by
the script.

Model information dumper usage
------------------------------

The basic usage is to run the script like this:

```sh
./info_dumper.py --all
```

This will print to standard output information about all models.

The only options accepted by the script are those described in the "Shared options"
section.

The script's output is a JSON array, each element of which is a JSON object
describing a single model. Each such object has the following keys:

* `name`: the identifier of the model, as accepted by the `--name` option.

* `description`: text describing the model. Paragraphs are separated by line feed characters.

* `framework`: a string identifying the framework whose format the model is downloaded in.
  Current possible values are `dldt` (Inference Engine IR), `caffe`, `caffe2`, `mxnet`, `onnx`,
  `pytorch` and `tf` (TensorFlow). Additional possible values might be added in the future.

* `license_url`: an URL for the license that the model is distributed under.

* `precisions`: the list of precisions that the model has IR files for. For models downloaded
  in a format other than the Inference Engine IR format, these are the precisions that the model
  converter can produce IR files in. Current possible values are:

  * `FP16`
  * `FP16-INT1`
  * `FP16-INT8`
  * `FP32`
  * `FP32-INT1`
  * `FP32-INT8`
  * `INT1`
  * `INT8`

  Additional possible values might be added in the future.

* `subdirectory`: the subdirectory of the output tree into which the downloaded or converted files
  will be placed by the downloader or the converter, respectively.

* `task_type`: a string identifying the type of task that the model performs. Current possible values
  are:

  * `action_recognition`
  * `classification`
  * `detection`
  * `face_recognition`
  * `feature_extraction`
  * `head_pose_estimation`
  * `human_pose_estimation`
  * `image_inpainting`
  * `image_processing`
  * `instance_segmentation`
  * `monocular_depth_estimation`
  * `object_attributes`
  * `optical_character_recognition`
  * `question_answering`
  * `semantic_segmentation`
  * `style_transfer`

  Additional possible values might be added in the future.

Shared options
--------------

The are certain options that all tools accept.

`-h`/`--help` can be used to print a help message:

```sh
./TOOL.py --help
```

There are several mutually exclusive filter options that select the models the
tool will process:

* `--all` selects all models.

  ```sh
  ./TOOL.py --all
  ```

* `--name` takes a comma-separated list of patterns and selects models that match
  at least one of these patterns. The patterns may contain shell-style wildcards.

  ```sh
  ./TOOL.py --name 'mtcnn-p,densenet-*'
  ```

  See https://docs.python.org/3/library/fnmatch.html for a full description of
  the pattern syntax.

* `--list` takes a path to a file that must contain a list of patterns and
  selects models that match at least one of those patterns.

  ```sh
  ./TOOL.py --list my.lst
  ```

  The file must contain one pattern per line. The pattern syntax is the same
  as for the `--name` option. Blank lines and comments starting with `#` are
  ignored. For example:

  ```
  mtcnn-p
  densenet-* # get all DenseNet variants
  ```

To see the available models, you can use the `--print_all` option. When this
option is specified, the tool will print all model names defined in the
configuration file and exit:

```
$ ./TOOL.py --print_all
action-recognition-0001-decoder
action-recognition-0001-encoder
age-gender-recognition-retail-0013
driver-action-recognition-adas-0002-decoder
driver-action-recognition-adas-0002-encoder
emotions-recognition-retail-0003
face-detection-adas-0001
face-detection-adas-binary-0001
face-detection-retail-0004
face-detection-retail-0005
[...]
```

Either `--print_all` or one of the filter options must be specified.

__________

OpenVINO is a trademark of Intel Corporation or its subsidiaries in the U.S.
and/or other countries.


Copyright &copy; 2018-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
