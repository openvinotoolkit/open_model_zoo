#!/bin/bash

# Copyright (c) 2018 Intel Corporation
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}" 
}
trap 'error ${LINENO}' ERR

DEMOS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build OMZ demos. Please install it. \n\n"
    exit 1
fi

build_dir=$HOME/inference_engine_demos_build
mkdir -p $build_dir
cd $build_dir
cmake -DCMAKE_BUILD_TYPE=Release $DEMOS_PATH
make -j8

printf "\nBuild completed, you can find binaries for all demos in the $HOME/inference_engine_demos_build/intel64/Release subfolder.\n\n"
