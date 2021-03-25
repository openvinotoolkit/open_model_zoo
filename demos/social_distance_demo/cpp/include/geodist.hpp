// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>

std::tuple<bool, bool, double> social_distance(std::tuple<int, int> &frame_shape,
                                               std::tuple<int, int> &a, std::tuple<int, int> &b,
                                               std::tuple<int, int> &c, std::tuple<int, int> &d,
                                               unsigned min_iter = 3, double min_w = 0, double max_w = 0);

std::tuple<int, int, int, int> get_crop(std::tuple<int, int, int, int> a, std::tuple<int, int, int, int> b);
