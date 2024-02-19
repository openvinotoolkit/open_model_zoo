#!/usr/bin/env python3
'''
 Copyright (C) 2021-2024 Intel Corporation

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
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gesture_storage',
                    help='Path to the gesture directory')

parser.add_argument('--classes_map',
                    help='Path to the classes file')

args = parser.parse_args()

with open(args.classes_map) as json_file:
    data = json.load(json_file)

dir = args.gesture_storage
files_list = []
for name in data:
    list = glob.glob(dir + name + '.mp4') + glob.glob(dir + name + '.avi')
    if len(list):
        files_list.append(list[0])

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

with open('gesture_gallery.json', 'w') as outfile:
    json.dump(objects, outfile, indent=4)
