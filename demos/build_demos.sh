#!/usr/bin/env bash

# Copyright (C) 2018-2024 Intel Corporation
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

usage() {
    echo "Build OpenVINO Runtime demos"
    echo
    echo "Options:"
    echo "  -h, --help                                      Print the help message"
    echo "  -b=DEMOS_BUILD_DIR, --build_dir=DEMOS_BUILD_DIR Specify the demo build directory"
    echo "  -DENABLE_PYTHON=y                               Whether to build extension modules for Python demos"
    echo '  --target=TARGETS                                A space sepparated list of demos to build. To build more than one specific demo use quotation marks ex. --target="classification_demo segmentation_demo"'
    echo
    exit 1
}

build_dir=$HOME/omz_demos_build
extra_cmake_opts=()
build_targets=()

for opt in "$@"; do
    case "$opt" in
    -h | --help)
        usage
        ;;
    -b=* | --build_dir=*)
        build_dir=("${opt#*=}")
        ;;
    -DENABLE_PYTHON=*)
        extra_cmake_opts+=("$opt")
        ;;
    --target=*)
        build_targets+=("${opt//=/ }")
        ;;
    *)
        printf "Unknown option: %q\n" "$opt"
        usage
        ;;
    esac
done

DEMOS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

printf "\nSetting environment variables for building demos...\n"

if [ -z "$INTEL_OPENVINO_DIR" ]; then
    if [ -e "$DEMOS_PATH/../../setupvars.sh" ]; then
        setvars_path="$DEMOS_PATH/../../setupvars.sh"
    elif [ -e "$DEMOS_PATH/../../../setupvars.sh" ]; then
        setvars_path="$DEMOS_PATH/../../../setupvars.sh"
    else
        printf "Error: Failed to set the environment variables automatically. To fix, run the following command:\n source <INSTALL_DIR>/bin/setupvars.sh\n where INSTALL_DIR is the OpenVINO installation directory.\n\n"
        exit 1
    fi
    if ! source "$setvars_path"; then
        printf "Unable to run ./setupvars.sh. Please check its presence. \n\n"
        exit 1
    fi
else
    # case for run with `sudo -E`
    source "$INTEL_OPENVINO_DIR/setupvars.sh"
fi

# CentOS 7 has two packages: cmake of version 2.8 and cmake3. install_openvino_dependencies.sh installs cmake3
if command -v cmake3 &>/dev/null; then
    CMAKE_EXEC=cmake3
elif command -v cmake &>/dev/null; then
    CMAKE_EXEC=cmake
else
    printf "\n\nCMAKE is not installed. It is required to build Open Model Zoo demos. Please install it. \n\n"
    exit 1
fi

OS_PATH=$(uname -m)
NUM_THREADS="-j2"

if [ "$OS_PATH" == "x86_64" ]; then
    OS_PATH="intel64"
    NUM_THREADS="-j8"
fi

if [ -e "$build_dir/CMakeCache.txt" ]; then
    rm -rf "$build_dir/CMakeCache.txt"
fi
mkdir -p "$build_dir"

(cd "$build_dir" && $CMAKE_EXEC -DCMAKE_BUILD_TYPE=Release "${extra_cmake_opts[@]}" "$DEMOS_PATH")
$CMAKE_EXEC --build "$build_dir" ${build_targets[@]} -- "$NUM_THREADS"

printf "\nBuild completed, you can find binaries for all demos in the %s subfolder.\n\n" "$build_dir/$OS_PATH/Release"
