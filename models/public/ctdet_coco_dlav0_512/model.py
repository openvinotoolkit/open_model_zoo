# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from opts import opts
from models.model import create_model, load_model

def ctdet_dlav0_34(weights_path):
    opt = opts().init(['ctdet', '--arch', 'dlav0_34'])

    return load_model(
        create_model(opt.arch, opt.heads, opt.head_conv),
        weights_path)
