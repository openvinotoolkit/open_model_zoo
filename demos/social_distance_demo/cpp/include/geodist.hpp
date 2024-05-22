// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <tuple>
#include <opencv2/core.hpp>

std::tuple<bool, bool, double> socialDistance(std::tuple<int, int>& frameShape,
                                               cv::Point2d& A, cv::Point2d& B,
                                               cv::Point2d& C, cv::Point2d& D,
                                               unsigned minIter = 3, double minW = 0, double maxW = 0);
