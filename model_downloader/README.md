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


      usage: downloader.py [-h] [-c CONFIG] [--name NAME] [--print_all]
                           [-o OUTPUT_DIR]

      optional arguments:
        -h, --help            show this help message and exit
        -c CONFIG, --config CONFIG
                              path to YML configuration file
        --name NAME           name of topology for downloading
        --print_all           print all available topologies
        -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                              path where to save topologies

      list_topologies.yml - default configuration file
   ```

*  Run the script with the default configuration file:

   ```sh
   ./downloader.py
   ```   
   or with a custom configuration file:
   
   ```sh   
   ./downloader.py -c <path_to_configuration_file>
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
   ```

*  Download only one topology (mtcnn-p in the following code example):
   
   ```sh
   ./downloader.py --name mtcnn-p
   ```

Expected free space to download all the topologies with the default configuration file is around 3.4 GB.

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
