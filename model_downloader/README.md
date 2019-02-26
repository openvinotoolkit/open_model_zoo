Public Topologies Downloader
============================

The script is designed to download popular public deep learning topologies and prepare models for the Model Optimizer tool.

Prerequisites
-------------

1. Install python3 (version 3.5.2 or higher) 
2. Install yaml and requests modules with the command

```sh
sudo -E pip3 install pyyaml requests
```   

Usage
-----

*  Run the script with `-h` key to see the help message:

   ```sh  
   ./downloader.py -h


      usage: downloader.py [-h] [-c CONFIG.YML] [--name PAT[,PAT...]]
                           [--list FILE.LST] [--all] [--print_all] [-o DIR]
                           [--cache_dir DIR]

      optional arguments:
        -h, --help            show this help message and exit
        -c CONFIG.YML, --config CONFIG.YML
                              path to YML configuration file
        --name PAT[,PAT...]   download only topologies whose names match at least
                              one of the specified patterns
        --list FILE.LST       download only topologies whose names match at least
                              one of the patterns in the specified file
        --all                 download all topologies from the configuration file
        --print_all           print all available topologies
        -o DIR, --output_dir DIR
                              path where to save topologies
        --cache_dir DIR       directory to use as a cache for downloaded files

      list_topologies.yml - default configuration file
   ```

*  Run the script with the default configuration file:

   ```sh
   ./downloader.py --all
   ```   
   or with a custom configuration file:
   
   ```sh   
   ./downloader.py --all -c <path_to_configuration_file>
   ```

*  Run the script with the `--print_all` option to see the available topologies:

   ```sh
   ./downloader.py --print_all

   densenet-121
   densenet-161
   densenet-169
   densenet-201
   squeezenet1.0
   squeezenet1.1
   mtcnn-p
   mtcnn-r
   mtcnn-o
   mobilenet-ssd
   vgg19
   vgg16
   ssd512
   ssd300
   inception-resnet-v2
   dilation
   googlenet-v1
   googlenet-v2
   googlenet-v4
   alexnet
   ssd_mobilenet_v2_coco
   resnet-50
   resnet-101
   resnet-152
   googlenet-v3
   se-inception
   se-resnet-101
   se-resnet-152
   se-resnet-50
   se-resnext-50
   se-resnext-101
   Sphereface
   license-plate-recognition-barrier-0007
   mobilenet-v1-1.0-224
   mobilenet-v2
   faster_rcnn_inception_v2_coco
   deeplabv3
   ctpn
   ssd_mobilenet_v1_coco
   faster_rcnn_resnet101_coco
   age-gender-recognition-retail-0013
   age-gender-recognition-retail-0013-fp16
   emotions-recognition-retail-0003
   emotions-recognition-retail-0003-fp16
   face-detection-adas-0001
   face-detection-adas-0001-fp16
   face-detection-retail-0004
   face-detection-retail-0004-fp16
   face-person-detection-retail-0002
   face-person-detection-retail-0002-fp16
   face-reidentification-retail-0071
   face-reidentification-retail-0071-fp16
   facial-landmarks-35-adas-0001
   facial-landmarks-35-adas-0001-fp16
   head-pose-estimation-adas-0001
   head-pose-estimation-adas-0001-fp16
   human-pose-estimation-0001
   human-pose-estimation-0001-fp16
   landmarks-regression-retail-0009
   landmarks-regression-retail-0009-fp16
   license-plate-recognition-barrier-0001
   license-plate-recognition-barrier-0001-fp16
   pedestrian-and-vehicle-detector-adas-0001
   pedestrian-and-vehicle-detector-adas-0001-fp16
   pedestrian-detection-adas-0002
   pedestrian-detection-adas-0002-fp16
   person-attributes-recognition-crossroad-0031
   person-attributes-recognition-crossroad-0031-fp16
   person-detection-action-recognition-0003
   person-detection-action-recognition-0003-fp16
   person-detection-retail-0001
   person-detection-retail-0001-fp16
   person-detection-retail-0013
   person-detection-retail-0013-fp16
   person-reidentification-retail-0031
   person-reidentification-retail-0031-fp16
   person-reidentification-retail-0076
   person-reidentification-retail-0076-fp16
   person-reidentification-retail-0079
   person-reidentification-retail-0079-fp16
   person-vehicle-bike-detection-crossroad-0078
   person-vehicle-bike-detection-crossroad-0078-fp16
   road-segmentation-adas-0001
   road-segmentation-adas-0001-fp16
   semantic-segmentation-adas-0001
   semantic-segmentation-adas-0001-fp16
   single-image-super-resolution-0034
   single-image-super-resolution-0034-fp16
   vehicle-attributes-recognition-barrier-0039
   vehicle-attributes-recognition-barrier-0039-fp16
   vehicle-detection-adas-0002
   vehicle-detection-adas-0002-fp16
   vehicle-license-plate-detection-barrier-0106
   vehicle-license-plate-detection-barrier-0106-fp16
   ```

*  Download only some topologies (mtcnn-p and all topologies starting with "densenet-" in the following code example):

   ```sh
   ./downloader.py --name 'mtcnn-p,densenet-*'
   ```

   The argument to `--name` must be a comma-separated list of patterns, which may contain shell-style wildcards.
   See https://docs.python.org/3/library/fnmatch.html for a full description of the pattern syntax.

   Alternatively, you can get the list of patterns from a file:

   ```sh
   ./downloader.py --list my.lst
   ```

   The specified file must list one pattern per line. Blank lines and comments starting with `#` will be ignored.
   For example:

   ```
   mtcnn-p
   densenet-* # get all DenseNet variants
   ```

Expected free space to download all the topologies with the default configuration file is around 7.09 GB.

__________

Copyright &copy; 2018 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
