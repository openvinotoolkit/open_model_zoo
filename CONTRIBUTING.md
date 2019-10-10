# How to contribute model to Open Model Zoo

We appreciate your intention to contribute model to OpenVINO&trade; Open Model Zoo (OMZ). This guide would help you and explain main issues. OMZ is licensed under the Apache License, Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms. Please note, that we accept models under permissive licenses, as **MIT**, **Apache 2.0**, **BSD-3-Clause**, etc. Otherwise, it may take longer time to get approve (or even refuse) for your model.

Nowadays OMZ supports models from frameworks: 
* Caffe\*
* Caffe2\* (via conversion to ONNX\*)
* TensorFlow\*
* PyTorch\* (via conversion to ONNX\*)
* MXNet\*

## Pull request requirements

Contribution to OMZ comes down to creating pull request (PR) in this repository. Please use `develop` branch when creating your PR. Pull request is strictly formalized and must contain:
* configuration file `model.yml` (learn more in [Configuration file](#configuration-file) section)
* documentation of model in markdown format (learn more in [Documentation](#documentation) section)
* accuracy validation configuration file (learn more in [Accuracy Validation](#accuracy-validation) section)
* license added to [tools/downloader/license.txt](tools/downloader/license.txt)
* (*optional*) demo (learn more about it in [Demo](#demo) section)

Name your model in OMZ according to the following rules:
- name must be consistent with original name, but complete match is not necessary
- use lowercase
- spaces are not allowed in the name, use `-` or `_` (`-` is preferable) as delimiters instead
- suffix to model name, according to origin framework (see **`framework`** description in [configuration file](#configuration-file) section), if you adding reimplementation of existing model in OMZ from another framework

This name will be used for downloading, converting, etc.
Example:
```
resnet-50-pytorch
mobilenet-v2-1.0-224
```

Files location:
* the configuration and documentation files must be in the `models/public/<model_name>` directory
* the validation configuration file must be in the `tools/accuracy_checker/configs` directory
* the demo must be in the `demos` directory

This PR must pass next tests:
* model is downloadable by `tools/downloader/downloader.py` script (see [Configuration file](#configuration-file) for details)
* model is convertible by `tools/downloader/converter.py` script (see [Model conversion](#model-conversion) for details)
* model can be used by demo or sample and provides adequate results (see [Demo](#demo) for details)
* model passes accuracy validation (see [Accuracy validation](#accuracy-validation) for details)

At the end, your PR will be reviewed by OMZ maintainers for consistence and legal compliance.

Your PR can be rejected in some cases, e.g.:
* inappropriate license (e.g. GPL-like licenses)
* inaccessible dataset
* PR fails one of the test above

## Configuration file

The model configuration file contains information about model: what it is, how to download it and how to convert it to IR format. This information must be specified in `model.yml` file, which must be located in the model subfolder. Let's look closer to the file content.

**`description`**

Description of the model. Must match with the description from model [documentation](#documentation).

**`task_type`**

Model task class:
- `action_recognition`
- `classification`
- `detection`
- `face_recognition`
- `head_pose_estimation`
- `human_pose_estimation`
- `image_processing`
- `instance_segmentation`
- `object_attributes`
- `optical_character_recognition`
- `semantic_segmentation`

If the task, that your model solves, is not listed here, please add new it to the list `KNOWN_TASK_TYPES` of the [tools/downloader/common.py](tools/downloader/common.py) file.

**`files`**

> Before filling this section, make sure that the model can be downloaded either via the direct HTTP(S) link or from Google Drive\*.

Downlodable files. Each file is described by:

* `name` - sets file name after downloading
* `size` - sets file size
* `sha256`  - sets file hash sum
* `source` - sets direct link to file *OR* describes file access parameters

> You may obtain hash sum using `sha256sum <file_name>` command on Linux\*.
 
If file is located on Google Drive\*, section `source` must contain:
- `$type: google_drive`
- `id` file ID on Google Drive\*

> **NOTE:** if file is on GitHub\*, use the specific file version.

**`postprocessing`** (*optional*)

Post processing of the downloaded files.

For unpacking archive:
- `$type: unpack_archive`
- `file` archive file name
- `format` archive format (zip | tar | gztar | bztar | xztar)

For replacement operation:
- `$type: regex_replace`
- `file` name of file where replacement must be executed
- `pattern` regular expression ([learn more](https://docs.python.org/3/library/re.html))
- `replacement` replacement string
- `count` (*optional*)  maximum number of pattern occurrences to be replaced

**`conversion_to_onnx_args`** (*optional*)

List of onnx conversion parameters, see `model_optimizer_args` for details. Applicable for Caffe2\* and PyTorch\* frameworks.

**`model_optimizer_args`**

Conversion parameters (learn more in [Model conversion](#model-conversion) section), e.g.:
```
  - --input=data
  - --mean_values=data[127.5]
  - --scale_values=data[127.5]
  - --reverse_input_channels
  - --output=prob
  - --input_model=$conv_dir/googlenet-v3.onnx
```
> **NOTE:** no need to specify `framework`, `data_type`, `model_name` and `output_dir`, since they are deduced automatically.

**`framework`**

Framework of the original model (`caffe`, `dldt`, `mxnet`, `pytorch`, `tf`, etc.).

**`license`**

Path to the model license.

### Example

In this [example](models/public/densenet-121-tf/model.yml) classification model DenseNet-121\*, pretrained in TensorFlow\*, is downloading from Google Drive\* as archive.

```
description: >-
  This is an Tensorflow\* version of `densenet-121` model, one of the DenseNet
  group of models designed to perform image classification. The weights were converted
  from DenseNet-Keras Models. For details see repository <https://github.com/pudae/tensorflow-densenet/>,
  paper <https://arxiv.org/pdf/1608.06993.pdf>
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
----
*After this step you will obtain **model.yml** file*

## Model conversion

Deep Learning Inference Engine (IE) supports models in the Intermediate Representation (IR) format. A model from any supported framework can be converted to IR using Model Optimizer tool included in the OpenVINO&trade; toolkit. Find more information about conversion in the [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html). After successful conversion you will get model in IR format `*.xml` representing net graph and `*.bin` containing net parameters. 

> **NOTE 1**: image pre-processing parameters (mean and scale) should be built into converted model to simplify model usage.

> **NOTE 2**: if model input is a color image, color channel order should be `BGR`.

*After this step you`ll get **conversion parameters** for Model Optimizer.*

## Demo

A demo shows the main idea of how to infer a model using IE. If your model solves one of the tasks supported by Open Model Zoo, try to find an appropriate option from [demos](https://docs.openvinotoolkit.org/latest/_demos_README.html) or [samples](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html). Otherwise, you must provide your own demo (C++ or Python).

Demos are required to support the following keys:

-    `-i "<input>"`             Required. Input to process.
-    `-m "<path>"`              Required. Path to an .xml file with a trained model. If the demo uses several models at the same time, use other keys prefixed with `-m`.
-    `-d "<device>"`            Optional. Default is CPU.
-    `-no_show`                 Optional. Do not visualize inference results.

> Note: For Python is preferable to use `-` instead of `_` as word separators (e.g. `-no-show`)

Also you can add any other necessary parameters.

If you add new demo, please provide auto-testing support too:
- add demo launch parameters in [demos/tests/cases.py](demos/tests/cases.py)
- prepare list of input images in [demos/tests/image_sequences.py](demos/tests/image_sequences.py)

*After this step you'll get **demo** for your model (if no demo was available)*

## Accuracy validation

Accuracy validation can be performed by the [Accuracy Checker](./tools/accuracy_checker) tool. This tool can use IE to run converted model or original framework to run original model. Accuracy Checker supports lots of datasets, metrics and preprocessing options, what makes validation quite simple (if task is supported by tool). You need only create configuration file, which contain necessary parameters to do accuracy validation (specify dataset and annotation, pre and post processing parameters, accuracy metric to compute and so on). Find more details [here](./tools/accuracy_checker#testing-new-models).

If model uses dataset which is unsupported by Accuracy Checker, you also must provide link to it. Please notice this issue in PR description. Don't forget about dataset license too (see [above](#how-to-contribute-model-to-open-model-zoo)).

When the configuration file is ready, you must run Accuracy Checker to obtain metric results. If they match your results, that means conversion was successful and Accuracy Checker fully supports your model, metric and dataset. If no - recheck [conversion](#model-conversion) parameters or validation configuration file.

*After this step you will get accuracy validation configuration file - **<model_name>.yml***

### Example

Let use one of the files from `tools/accuracy_checker/configs`, for example, validation configuration file for [AlexNet](tools/accuracy_checker/configs/alexnet.yml):
```
models:
  - name: alexnet-cf
    launchers:
      - framework: caffe
        model:   public/alexnet/alexnet.prototxt
        weights: public/alexnet/alexnet.caffemodel
        adapter: classification
    datasets:
      - name: imagenet_1000_classes
        preprocessing:
          - type: resize
            size: 256
          - type: crop
            size: 227
          - type: normalization
            mean: 104, 117, 123

  - name: alexnet
    launchers:
      - framework: dlsdk
        tags:
          - FP32
        model:   public/alexnet/FP32/alexnet.xml
        weights: public/alexnet/FP32/alexnet.bin
        adapter: classification

      - framework: dlsdk
        tags:
          - FP16
        model:   public/alexnet/FP16/alexnet.xml
        weights: public/alexnet/FP16/alexnet.bin
        adapter: classification

    datasets:
      - name: imagenet_1000_classes
        preprocessing:
          - type: resize
            size: 256
          - type: crop
            size: 227

        metrics:
          - name: accuracy@top1
            type: accuracy
            top_k: 1
          - name: acciracy@top5
            type: accuracy
            top_k: 5
```


## Documentation

Documentation is very important part of model contribution, it helps to better understand possible use of the model. Documentation must be named after the name of the model.
The documentation should contain:
* description of model
	* main purpose
	* features
	* references to paper or/and source
* model specification
	* type
	* framework
	* GFLOPs (*if available*)
	* number of parameters (*if available*)
* validation dataset description and/or link
* main accuracy values (also description of metric)
* detailed description of input and output for original and converted models

Learn the detailed structure and headers naming convention from any model documentation, e.g. [alexnet](./models/public/alexnet/alexnet.md).

---
*After this step you will obtain **<model_name>.md** - documentation file*

## Legal Information

[\*] Other names and brands may be claimed as the property of others.

OpenVINO is a trademark of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

Copyright &copy; 2018-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
