#!/usr/bin/env python3

"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import re
import yaml
import time
import sys
import hashlib
import tarfile
import requests
import shutil
import argparse

def compare_hash(hash_of_gold, path_to_file):
    with open(path_to_file, 'rb') as f:
        hash_of_file = hashlib.sha256(f.read()).hexdigest()
        if hash_of_file != hash_of_gold :
            print('########## Hash sum of', path_to_file, 'differs from the target, the topology will be deleted !!! ##########')
            shutil.rmtree(os.path.dirname(path_to_file))

def process_download(chunk_size, response, size, file):
    start_time = time.monotonic()
    progress_size = 0
    for chunk in response.iter_content(chunk_size):
        if chunk:
            duration = time.monotonic() - start_time
            progress_size += len(chunk)
            if duration != 0:
                speed = progress_size // (1024 * duration)
                percent = min(progress_size * 100 // size, 100)
                sys.stdout.write('\r...%d%%, %d KB, %d KB/s, %d seconds passed ========= ' %
                                (percent, progress_size / 1024, speed, duration))
                sys.stdout.flush()
            file.write(chunk)

def download(url, path, name, total_size = 0):
    destination = os.path.join(path, name)
    chunk_size = 8192
    with requests.Session() as session, open(destination, 'wb') as f:
        response = session.get(url, stream = True)
        if total_size != 0:
            size = total_size
        else:
            size = int(response.headers.get('content-length', 0))
        process_download(chunk_size, response, size, f)
    print(name, '====>', destination)
    print('')

def get_extensions(framework):
    extensions = []
    if framework == 'caffe':
        extensions = ['.prototxt', '.caffemodel']
    elif framework == 'tf':
        extensions = ['.prototxt', '.frozen.pb']
    elif framework == 'mxnet':
        extensions = ['.json', '.params']
    elif framework == 'dldt':
        extensions = ['.xml', '.bin']
    else:
        sys.exit('Unknown framework: "{}"'.format(framework))
    return extensions

def download_file_from_google_drive(id, path, name, total_size):
    URL = 'https://docs.google.com/uc?export=download'
    with requests.Session() as session:
        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)
        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        save_response_content(response, path, name, total_size)
    print('')

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, path, name, size):
    destination = os.path.join(path, name)
    chunk_size = 32768
    with open(destination, 'wb') as f:
        process_download(chunk_size, response, size, f)
    print(name, '====>', destination)

def delete_param(model):
    tmpfile = os.path.join(args.output_dir, 'tmp.txt')
    with open(model, 'r') as input_file, open(tmpfile, 'w') as output_file:
        data=input_file.read()
        updated_data = re.sub(' +save_output_param \{.*\n.*\n +\}\n', '', data, count=1)
        output_file.write(updated_data)
    shutil.move(tmpfile, model)

def layers_to_layer(model):
    tmpfile = os.path.join(args.output_dir, 'tmp.txt')
    with open(model, 'r') as input_file, open(tmpfile, 'w') as output_file:
        data=input_file.read()
        updated_data = data.replace('layers {', 'layer {')
        output_file.write(updated_data)
    shutil.move(tmpfile, model)

def change_dim(model, old_dim, new_dim):
    new = 'dim: ' + str(new_dim)
    old = 'dim: ' + str(old_dim)
    tmpfile = os.path.join(args.output_dir, 'tmp.txt')
    with open(model, 'r') as input_file, open(tmpfile, 'w') as output_file:
        data=input_file.read()
        data = data.replace(old, new, 1)
        output_file.write(data)
    shutil.move(tmpfile, model)

parser = argparse.ArgumentParser(epilog = 'list_topologies.yml - default configuration file')
parser.add_argument('-c', '--config', type = str, default = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'list_topologies.yml'), help = 'path to YML configuration file')
parser.add_argument('--name', help = 'name of topology for downloading')
parser.add_argument('--print_all', action = 'store_true', help = 'print all available topologies')
parser.add_argument('-o', '--output_dir' , type = str, default = os.getcwd(), help = 'path where to save topologies')
args = parser.parse_args()
path_to_config = args.config

with open(path_to_config) as stream:
    try:
        c_new = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit('Cannot parse the YML, please check the configuration file')
if  args.print_all:
    for top in c_new['topologies']:
        print(top['name'])
    sys.exit()
if args.name != None:
    try:
        topologies = next([top] for top in c_new['topologies'] if top['name'] == args.name)
    except StopIteration:
        sys.exit('No such topology: "{}"'.format(args.name))
else:
    topologies = c_new['topologies']

print('')
print('###############|| Start downloading models ||###############')
print('')
for top in topologies:
    output = os.path.join(args.output_dir, top['output'])
    os.makedirs(output, exist_ok=True)
    if {'model_google_drive_id', 'model_size'} <= top.keys():
        download_file_from_google_drive(top['model_google_drive_id'], output, top['name'] + get_extensions(top['framework'])[0], top['model_size'])
    elif 'model_size' in top:
        download(top['model'], output, top['name'] + get_extensions(top['framework'])[0], top['model_size'])
    elif 'model' in top:
        download(top['model'], output, top['name'] + get_extensions(top['framework'])[0])
print('###############|| Start downloading weights ||###############')
print('')
for top in topologies:
    output = os.path.join(args.output_dir, top['output'])
    if {'weights_google_drive_id', 'weights_size'} <= top.keys():
        download_file_from_google_drive(top['weights_google_drive_id'], output, top['name'] + get_extensions(top['framework'])[1], top['weights_size'])
    elif 'weights_size' in top:
        download(top['weights'], output, top['name']+ get_extensions(top['framework'])[1], top['weights_size'])
    elif 'weights' in top:
        download(top['weights'], output, top['name']+ get_extensions(top['framework'])[1])
print('###############|| Start downloading topologies in tarballs ||###############')
print('')
for top in topologies:
    output = os.path.join(args.output_dir, top['output'])
    if {'tar_google_drive_id', 'tar_size'} <= top.keys():
        download_file_from_google_drive(top['tar_google_drive_id'], output, top['name']+'.tar.gz', top['tar_size'])
    elif 'tar' in top:
        download(top['tar'], output, top['name']+'.tar.gz')
print('')
print('###############|| Post processing ||###############')
print('')
for top in topologies:
    model_name = top['name'] + get_extensions(top['framework'])[0]
    weights_name = top['name'] + get_extensions(top['framework'])[1]
    output = os.path.join(args.output_dir, top['output'])
    path_to_model = os.path.join(output, model_name)
    path_to_weights = os.path.join(output, weights_name)
    for path, subdirs, files in os.walk(output):
        for name in files:
            fname = os.path.join(path, name)
            if fname.endswith('.tar.gz'):
                print('========= Extracting files from %s.tar.gz' % (top['name']))
                shutil.unpack_archive(fname, path)
    if {'model_path_prefix', 'weights_path_prefix'} <= top.keys():
        downloaded_model = os.path.join(output, top['model_path_prefix'])
        downloaded_weights = os.path.join(output, top['weights_path_prefix'])
        if (os.path.exists(downloaded_model) and os.path.exists(downloaded_weights)):
            print('========= Moving %s and %s to %s after untarring the archive =========' % (model_name, weights_name, output))
            shutil.move(downloaded_model, path_to_model)
            shutil.move(downloaded_weights, path_to_weights)
    elif 'model_path_prefix' in top:
        downloaded_model = os.path.join(output, top['model_path_prefix'])
        if os.path.exists(downloaded_model):
            print('========= Moving %s to %s after untarring the archive =========' % (weights_name, output))
            shutil.move(downloaded_model, path_to_weights)
    if 'model_hash' in top:
        if os.path.exists(path_to_model):
            compare_hash(top['model_hash'], path_to_model)
    if 'weights_hash' in top:
        if os.path.exists(path_to_weights):
            compare_hash(top['weights_hash'], path_to_weights)
    if 'delete_output_param' in top:
        if os.path.exists(path_to_model):
            print('========= Deleting "save_output_param" from %s =========' % (model_name))
            delete_param(path_to_model)
    if {'old_dims', 'new_dims'} <= top.keys():
        if os.path.exists(path_to_model):
            print('========= Changing input dimensions in %s =========' % (model_name))
            for j in range(len(top['old_dims'])):
                change_dim(path_to_model, top['old_dims'][j], top['new_dims'][j])
    if 'layers_to_layer' in top:
        if os.path.exists(path_to_model):
            print('========= Moving to new Caffe layer presentation %s =========' % (model_name))
            layers_to_layer(path_to_model)
