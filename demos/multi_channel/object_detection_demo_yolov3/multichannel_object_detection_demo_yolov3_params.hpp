// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Optional. Probability threshold for detections";

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter. Ignored for human-pose-estimation
DEFINE_double(t, 0.5, thresh_output_message);
