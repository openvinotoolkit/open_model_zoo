// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>

#define DEFINE_INPUT_FLAGS \
DEFINE_string(i, "", input_message); \
DEFINE_bool(loop, false, loop_message);

#define DEFINE_OUTPUT_FLAGS \
DEFINE_string(o, "", output_message); \
DEFINE_uint32(limit, 1000, limit_message);

static const char input_message[] = "Required. An input to process. The input must be a single image, a folder of "
    "images, video file or camera id.";
static const char loop_message[] = "Optional. Enable reading the input in a loop.";
static const char output_message[] = "Optional. Name of the output file(s) to save.";
static const char limit_message[] = "Optional. Number of frames to store in output. If 0 is set, all frames are stored.";
