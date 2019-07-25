Model Downloader and other automation tools
===========================================

This directory contains scripts that automate certain model-related tasks
based on the included configuration file.

* `downloader.py` (model downloader) downloads model files from online sources
  and, if necessary, patches them to make them more usable with Model
  Optimizer;

* `converter.py` (model converter) converts the models that are not in the
  Inference Engine IR format into that format using Model Optimizer.

* `info_dumper.py` (model information dumper) prints information about the models
  in a stable machine-readable format.

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

Model downloader usage
----------------------

The basic usage is to run the script like this:

```sh
./downloader.py --all
```

This will download all models into a directory tree rooted in the current
directory. To download into a different directory, use the `-o`/`--output_dir`
option:

```sh
./downloader.py --all --output_dir my/download/directory
```

The `--all` option can be replaced with other filter options to download only
a subset of models. See the "Shared options" section.

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

See the "Shared options" section for information on other options accepted by
the script.

Model converter usage
---------------------

The basic usage is to run the script like this:

```sh
./converter.py --all
```

This will convert all models into the Inference Engine IR format. Models that
were originally in that format are ignored.

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

The `--all` option can be replaced with other filter options to convert only
a subset of models. See the "Shared options" section.

The script will attempt to locate Model Optimizer using the environment
variables set by the OpenVINO&trade; toolkit's `setupvars.sh`/`setupvars.bat`
script. You can override this heuristic with the `--mo` option:

```sh
./converter.py --all --mo my/openvino/path/model_optimizer/mo.py
```

By default, the script will run Model Optimizer using the same Python executable
that was used to run the script itself. To use a different Python executable,
use the `-p`/`--python` option:

```sh
./converter.py --all --python my/python
```

The script can print the conversion commands without actually running them.
To do this, use the `--dry-run` option:

```sh
./converter.py --all --dry-run
```

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
  Current possible values are `dldt` (Inference Engine IR), `caffe`, `mxnet` and `tf` (TensorFlow).
  Additional possible values might be added in the future.

* `license_url`: an URL for the license that the model is distributed under.

* `precisions`: the list of precisions that the model has IR files for. For models downloaded
  in a format other than the Inference Engine IR format, these are the precisions that the model
  converter can produce IR files in. Current possible values are `FP16`, `FP32`, `INT1`, `INT8`;
  more might be added in the future.

* `subdirectory`: the subdirectory of the output tree into which the downloaded or converted files
  will be placed by the downloader or the converter, respectively.

* `task_type`: a string identifying the type of task that the model performs. Current possible values
  are:

  * `action_recognition`
  * `classification`
  * `detection`
  * `face_recognition`
  * `head_pose_estimation`
  * `human_pose_estimation`
  * `image_processing`
  * `instance_segmentation`
  * `object_attributes`
  * `optical_character_recognition`
  * `semantic_segmentation`

  Additional possible values might be added in the future.

Shared options
--------------

The are certain options that both tools accept.

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
Sphereface
action-recognition-0001-decoder
action-recognition-0001-encoder
age-gender-recognition-retail-0013
alexnet
brain-tumor-segmentation-0001
ctpn
deeplabv3
densenet-121
densenet-121-tf
[...]
```

Either `--print_all` or one of the filter options must be specified.

By default, the tools will get information about the models from the configuration
file in the automation tool directory. You can use a custom configuration file
instead with the `-c`/`--config` option:

```sh
./TOOL.py --all --config my-config.yml
```

Note, however, that the configuration file format is currently undocumented and
may change in incompatible ways in future versions.

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
