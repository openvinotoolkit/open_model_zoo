// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace custom {
size_t getTopLeftPointIdx(const std::vector<cv::Point2f>& points);
}
