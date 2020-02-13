#!/usr/bin/env python
'''
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import glob
import os
import json
import sys

if len(sys.argv) == 2:
    dir = sys.argv[1]
else:
    dir = '.' + os.sep

files_list = glob.glob(dir + '*.png') + glob.glob(dir + '*.jpg')

labels = []
objects = {}

for file in files_list:
    label = file.rpartition(os.sep)[2].rpartition('.')[0]
    path = os.path.abspath(file)

    if label in labels:
        raise Exception('An item with the label {} already exists in the gallery!'.format(label))
    else:
        labels.append(label)
        objects[label] = [path]

with open('faces_gallery.json', 'w') as outfile:
    json.dump(objects, outfile, indent=4)
