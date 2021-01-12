// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>

#define DEFINE_INPUT_FLAGS \
DEFINE_string(i, "", input_message); \
DEFINE_bool(loop, false, loop_message);

static const char input_message[] = "Required. An input to process. The input must be a single image, a folder of "
    "images or anything that cv::VideoCapture can process.";
static const char loop_message[] = "Optional. Enable reading the input in a loop.";
