// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

static const char thresh_output_message[] = "Optional. Probability threshold for detections";

DEFINE_double(t, 0.5, thresh_output_message);
