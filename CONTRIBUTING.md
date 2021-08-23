# How to Contribute Models to Open Model Zoo

We appreciate your intention to contribute model to the OpenVINO&trade; Open Model Zoo (OMZ). OMZ is licensed under the Apache\* License, Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms. Note that we accept models under permissive licenses, such as **MIT**, **Apache 2.0**, and **BSD-3-Clause**. Otherwise, it might take longer time to get your model approved.

Frameworks supported by the Open Model Zoo:
* Caffe\*
* Caffe2\* (via conversion to ONNX\*)
* TensorFlow\*
* PyTorch\* (via conversion to ONNX\*)
* MXNet\*

Open Model Zoo also supports models already in the ONNX format.

## Pull Request Requirements

To contribute to OMZ, create a pull request (PR) in this repository using the `develop` branch.
Pull requests are strictly formalized and are reviewed by the OMZ maintainers for consistence and legal compliance.

Each PR contributing a model must contain:
* [configuration file `model.yml`](#configuration-file)
* [documentation of model in markdown format](#documentation)
* [accuracy validation configuration file](#accuracy-validation)
* (*optional*) [demo](#demo)

Follow the rules in the sections below before submitting a pull request.

### Model Name

Name your model in OMZ according to the following rules:
- Use a name that is consistent with an original name, but complete match is not necessary
- Use lowercase
- Use `-`(preferable) or `_` as delimiters, for spaces are not allowed
- Add a suffix according to framework identifier (see **`framework`** description in the [configuration file](#configuration-file) section for examples), if the model is a reimplementation of an existing model from another framework

This name will be used for downloading, converting, and other operations.
Examples of model names:
- `resnet-50-pytorch`
- `mobilenet-v2-1.0-224`

### Files Location

Place your files as shown in the table below:

File | Destination
---|---
configuration file | `models/public/<model_name>/model.yml`
documentation file | `models/public/<model_name>/README.md`
validation configuration file|`models/public/<model_name>/accuracy-check.yml`
demo|`demos/<demo_name>`<br>or<br>`demos/python_demos/<demo_name>`

### Tests

Your PR must pass next tests:
* Model is downloadable by the `tools/downloader/downloader.py` script. See [Configuration file](#configuration-file) for details.
* Model is convertible by the `tools/downloader/converter.py` script. See [Model conversion](#model-conversion) for details.
* Model is usable by demo or sample and provides adequate results. See [Demo](#demo) for details.
* Model passes accuracy validation. See [Accuracy validation](#accuracy-validation) for details.


### PR Rejection

Your PR may be rejected in some cases, for example:
* If a license is inappropriate (such as GPL-like licenses).
* If a dataset is inaccessible.
* If the PR fails one of the tests above.

## Configuration File

The model configuration file contains information about model: what it is, how to download it, and how to convert it to the IR format. This information must be specified in the `model.yml` file that must be located in the model subfolder.

The detailed descriptions of file entries provided below.

**`description`**

Description of the model. Must match with the description from the model [documentation](#documentation). Use [this](./ci/documentation_updater/documentation_updater.py) script for easy update.

**`task_type`**

[Model task type](tools/downloader/README.md#model-information-dumper-usage). If there is no task type of your model, add a new one to the list `KNOWN_TASK_TYPES` of the [`open_model_zoo.model_tools._common`](tools/downloader/src/open_model_zoo/model_tools/_common.py) module.

**`files`**

> **NOTE**: Before filling this section, make sure that the model can be downloaded either via a direct HTTP(S) link or from Google Drive\*.

Downloadable files. Each file is described by:

* `name` - sets a file name after downloading
* `size` - sets a file size
* `sha256`  - sets a file hash sum
* `source` - sets a direct link to a file *OR* describes a file access parameters

> **TIP**: You can obtain a hash sum using the `sha256sum <file_name>` command on Linux\*.

If file is located on Google Drive\*, the `source` section must contain:
- `$type: google_drive`
- `id` file ID on Google Drive\*

> **NOTE:** If file is on GitHub\*, use the specific file version.

**`postprocessing`** (*optional*)

Post processing of the downloaded files.

For unpacking archive:
- `$type: unpack_archive`
- `file` — Archive file name
- `format` — Archive format (zip | tar | gztar | bztar | xztar)

For replacement operation:
- `$type: regex_replace`
- `file` — Name of file to run replacement in
- `pattern` — [Regular expression](https://docs.python.org/3/library/re.html)
- `replacement` — Replacement string
- `count` (*optional*)  — Exact number of replacements (if number of `pattern` occurrences less then this number, downloading will be aborted)

**`conversion_to_onnx_args`** (*only for Caffe2\*, PyTorch\* models*)

List of ONNX\* conversion parameters, see `model_optimizer_args` for details.

**`model_optimizer_args`**

Conversion parameters (learn more in the [Model conversion](#model-conversion) section). For example:
```
  - --input=data
  - --mean_values=data[127.5]
  - --scale_values=data[127.5]
  - --reverse_input_channels
  - --output=prob
  - --input_model=$conv_dir/googlenet-v3.onnx
```

> **NOTE:** Do not specify `framework`, `data_type`, `model_name` and `output_dir`, since they are deduced automatically.

> **NOTE:** `$dl_dir` used to substitute subdirectory of downloaded model and `$conv_dir` used to substitute subdirectory of converted model.

**`framework`**

Framework of the original model (see [here](tools/downloader/README.md#model-information-dumper-usage) for details).

**`license`**

URL of the model license.

### Example

This example shows how to download the classification model [DenseNet-121*](models/public/densenet-121-tf/model.yml) pretrained in TensorFlow\* from Google Drive\* as an archive.

```
description: >-
  This is a TensorFlow\* version of `densenet-121` model, one of the DenseNet
  group of models designed to perform image classification. The weights were converted
  from DenseNet-Keras Models. For details see repository <https://github.com/pudae/tensorflow-densenet/>,
  paper <https://arxiv.org/abs/1608.06993>
task_type: classification
files:
  - name: tf-densenet121.tar.gz
    size: 30597420
    sha256: b31ec840358f1d20e1c6364d05ce463cb0bc0480042e663ad54547189501852d
    source:
      $type: google_drive
      id: 0B_fUSpodN0t0eW1sVk1aeWREaDA
postprocessing:
  - $type: unpack_archive
    format: gztar
    file: tf-densenet121.tar.gz
model_optimizer_args:
  - --reverse_input_channels
  - --input_shape=[1,224,224,3]
  - --input=Placeholder
  - --mean_values=Placeholder[123.68,116.78,103.94]
  - --scale_values=Placeholder[58.8235294117647]
  - --output=densenet121/predictions/Reshape_1
  - --input_meta_graph=$dl_dir/tf-densenet121.ckpt.meta
framework: tf
license: https://raw.githubusercontent.com/pudae/tensorflow-densenet/master/LICENSE
```

## Model Conversion

Deep Learning Inference Engine (IE) supports models in the Intermediate Representation (IR) format. A model from any supported framework can be converted to IR using the Model Optimizer tool included in the OpenVINO&trade; toolkit. Find more information about conversion in the [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html). After a successful conversion you get a model in the IR format, with the `*.xml` file representing the net graph and the `*.bin` file containing the net parameters.

> **NOTE 1**: Image preprocessing parameters (mean and scale) must be built into a converted model to simplify model usage.

> **NOTE 2**: If a model input is a color image, color channel order should be `BGR`.

## Demo

A demo shows the main idea of how to infer a model using IE. If your model solves one of the tasks supported by the Open Model Zoo, try to find an appropriate option from [demos](demos/README.md) or [samples](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html). Otherwise, you must provide your own demo (C++ or Python).

The demo's name should end with `_demo` suffix to follow the convention of the project.

Demos are required to support the following keys:

 -  `-i "<input>"`: Required. An input to process. The input can usually be a single image, a folder of images or anything that OpenCV's `VideoCapture` can process.
 -  `-m "<path>"`: Required. Path to an .xml file with a trained model. If the demo uses several models at the same time, use other keys prefixed with `-m_`.
 -  `-d "<device>"`: Optional. Specifies a target device to infer on. CPU, GPU, HDDL or MYRIAD is acceptable. Default must be CPU. If the demo uses several models at the same time, use keys prefixed with `d_` (just like keys `m_*` above) to specify device for each model.
 -  `-no_show`: Optional. Do not visualize inference results.

> **TIP**: For Python, it is preferable to use `--` instead of `-` for long keys.

You can also add any other necessary parameters.

Add `README.md` file, which describes demo usage. Update [demos' README.md](demos/README.md) adding your demo to the list.

## Accuracy Validation

Accuracy validation can be performed by the [Accuracy Checker](./tools/accuracy_checker/README.md) tool. This tool can use either IE to run a converted model, or an original framework to run an original model. Accuracy Checker supports lots of datasets, metrics and preprocessing options, which simplifies validation if a task is supported by the tool. You only need to create a configuration file that contains necessary parameters for accuracy validation (specify a dataset and annotation, pre- and post-processing parameters, accuracy metrics to compute and so on) of converted model. For details, refer to [Testing new models](./tools/accuracy_checker/README.md#testing-new-models).

If a model uses a dataset which is not supported by the Accuracy Checker, you also must provide the license and the link to it and mention it in the PR description.

When the configuration file is ready, you must run the Accuracy Checker to obtain metric results. If they match your results, that means conversion was successful and the Accuracy Checker fully supports your model, metric and dataset. Otherwise, recheck the [conversion](#model-conversion) parameters or the validation configuration file.

### Example

This example uses validation configuration file for [DenseNet-121](models/public/densenet-121-tf/accuracy-check.yml)\* from TensorFlow\*:
```
models:
  - name: densenet-121-tf
    launchers:
      - framework: dlsdk
        adapter: classification

    datasets:
      - name: imagenet_1000_classes
        preprocessing:
          - type: resize
            size: 256
          - type: crop
            size: 224
```


## Documentation

Documentation is a very important part of model contribution as it helps to better understand the possible usage of the model. It must be located in a `README.md` file in the model subdirectory.
The documentation should contain:
* description of a model
	* main purpose
	* features
	* references to a paper or/and a source
* model specification
	* type
	* framework
	* GFLOPs (*if available*)
	* number of parameters (*if available*)
* validation dataset description and/or a link
* main accuracy values (also description of a metric)
* detailed description of input and output for original and converted models
* the model's licensing terms

Learn the detailed structure and headers naming convention from any model documentation (for example, [alexnet](./models/public/alexnet/README.md)).

## Legal Information

[\*] Other names and brands may be claimed as the property of others.

OpenVINO is a trademark of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

Copyright &copy; 2018-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
