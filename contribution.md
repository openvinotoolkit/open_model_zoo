# How to contribute to OMZ

From this document you will know how to contribute your model to OpenVINO&trade; Open Model Zoo. Almost any model from supported frameworks (see list below) can be added. To do this do next few steps.

1. [Model location](#model-location)
2. [Model conversion](#model-conversion)
3. [Demo](#demo)
4. [Accuracy validation](#accuracy-validation)
7. [Configuration file](#configuration-file)
6. [Documentation](#documentation)
7. [Pull request requirements](#pull-request-requirements)

List of supported frameworks:
* Caffe\*
* Caffe2\* (by conversion to ONNX\*)
* TensorFlow\*
* MXNet\*
* PyTorch\* (by conversion to ONNX\*)



## Model location

Upload your model to any Internet file storage with easy and direct access to it. It can be www.github.com, GoogleDrive\*, or any other.

*After this step you will get **links** to the model*

## Model conversion

OpenVINO&trade; supports models in its own format IR. Model from any supported framework can be easily converted to IR using Model Optimizer tool included in OpenVINO&trade; package. More information about conversion you can learn [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html). After successful conversion you will get model in IR format `*.xml` representing net graph and `*.bin` containing net parameters. 

> **NOTE 1**: due OpenVINO&trade; paradigms, mean and scale values are built-in converted model.

> **NOTE 2**: due OpenVINO&trade; paradigms, if model take colored image as input, color channel order supposed to be `BGR`.

*After this step you`ll get **conversion parameters** for Model Optimizer.*

## Demo

Demo will show main idea of how work with your model. If your model solves one of the supported by Open Model Zoo task, try find appropriate option from [demos](https://docs.openvinotoolkit.org/latest/_demos_README.html) or [samples](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html).

If appropriate demo or sample are absent, you must provide your own demo (C++ or Python). Demo's input requires next options:
```
    -i "<path>"              Optional. Path to a input file or directory (for multiple inferences). By default input must be generated randomly.
    -m "<path>"              Required. Path to an .xml file with a trained model
    -d "<device>"            Optional. Target device for model inference. Usually CPU and GPU. By default CPU
    -no_show                 Optional. Do not show inference result.
```
Also you can add all necessary parameters for inference your model.

## Accuracy validation

To run accuracy validation, use [Accuracy Checker](./tools/accuracy_checker/README.md) tool, provided with repository. Doing this very simple if model task from supported. You must only create accuracy validation configuration file in this case. 

When the configuration file is ready, you must run Accuracy Checker to obtain metric results. If they match your results, that means  conversion was fully successful and Accuracy Checker fully supports your model, metric and dataset. If no - recheck [conversion][#model-conversion] parameters or validation configuration.

*After this step you will get accuracy validation configuration file - **<model_name>.yml***

## Configuration file

Models configuration file contains information about model: what it is, how to download it and how to convert it to IR format. This information must be specified in `model.yml` file, which must be in the models subfolder. Let look closer to the file content.

**`description`**

This tag must contain description of model.

**`task_type`**

This tag describes on of the task that model solves:
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

If your model solves another task, you can freely add it with modification of [tools/downloader/common.py](tools/downloader/common.py) file list `KNOWN_TASK_TYPES`

**`files`**

You must describe all files, which must be downloaded, in this section. Each file must is described in few tags:

* `name` sets file name after downloading
* `size` sets file size
* `sha256` sets file hash sum
* `source` sets direct link to file *OR* describes file access parameters

If file is located on GoogleDrive\*, section `source` must contain:
```
 - $type: google_drive
   id: <file id>
```
**`postprocessing`** (*optional*)

Sometimes right after downloading model are not ready for conversion, or conversion may be incorrect or failure. It may be avoided by some manipulation with original files, such as unpacking, replacing or deleting some part of file. This manipulation must be described in this section.

For unpacking archive:

```
  - $type: unpack_archive
    file: <file name>
    format: zip | tar | gztar | bztar | xztar
```

For replacement operations:

```
  - $type: regex_replace
    file: 
    pattern:
    replacement:
    count: 
```
where
- `file` name of file where replacement must be executed
- `pattern` string or regexp ([learn more](https://docs.python.org/2/library/re.html)) to find
- `replacement` replacement string
- `count` (*optional*)  maximum number of pattern occurrences to be replaced


**`pytorch_to_onnx`** (*optional*)

List of pytorch-to-onnx conversion parameters, see `model_optimizer_args` for details.

**`caffe2_to_onnx`** (*optional*)

List of caffe2-to-onnx conversion parameters, see `model_optimizer_args` for details.

**`model_optimizer_args`**

Conversion parameter, obtained [earlier](#model-conversion), must be specified in this section, e.g.:
```
  - --input=data
  - --mean_values=data[127.5]
  - --scale_values=data[127.5]
  - --reverse_input_channels
  - --output=prob
  - --input_model=$conv_dir/googlenet-v3.onnx
```
> **NOTE:**  no need to specify `framework`, `data_type`, `model_name` and `output_dir` parameters since them are deduced automatically.

**`framework`**

Framework of original model (`caffe`, `dldt`, `mxnet`, `pytorch`, `tf`)

**`license`**

Path to model's license.

----
*After this step you will obtain **model.yml** file*

### Example

In this [example](models/public/densenet-121-tf/model.yml) classificational model DenseNet-121\*, pretrained in TensorFlow\*, is downloading from GoogleDrive\* as archive.

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

## Documentation

Documentation is very important part of model contribution, it helps to better understand possible usage of the model. Documentation must be named after suggested models name.
Documentation must contain:
* description of model
	* main purpose
	* features
	* links to paper or/and source
* model specification, e.g. type, source framework, GFLOPs and number of parameters
	* type
	* framework
	* GFLOPs
	* number of parameters
* main accuracy values (also description of metric)
* detailed description of input and output for original and converted models

Detailed structure and headers naming convention you can learn from any other model, e.g. [alexnet](./models/public/alexnet/alexnet.md).

---
*After this step you will obtain **<model_name>.md** - documentation file*

## Pull request requirements

Contribution to OpenVINO&trade; Open Model Zoo comes down to creating pull request in this repository. This pull request is strictly formalized and must contains changes:
* configuration file  - `model.yml` [from here](#configuration-file)
* documentation of model in markdown format [from here](#documentation)
* accuracy validation configuration file [from here](#accuracy-validation)
* license added to [tools/downloader/license.txt](tools/downloader/license.txt)

Configuration and documentation files must be located in `models/public` directory in subfolder, which name will represent model name in Open Model Zoo and will be used by downloader and converter tools. Also, please add suffix to model name, according to origin framework (e.g. `cf`, `cf2`, `tf`, `mx` or `pt`). 

Validation configuration file must be located in [tools/accuracy_checker/configs](tools/accuracy_checker/configs).

## Legal Information

[\*] Other names and brands may be claimed as the property of others.

OpenVINO is a trademark of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

Copyright &copy; 2018-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
