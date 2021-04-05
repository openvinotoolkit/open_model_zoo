// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>

std::tuple<bool, bool, double> socialDistance(std::tuple<int, int> &frameShape,
                                               std::tuple<int, int> &a, std::tuple<int, int> &b,
                                               std::tuple<int, int> &c, std::tuple<int, int> &d,
                                               unsigned minIter = 3, double minW = 0, double maxW = 0);

std::tuple<int, int, int, int> getCrop(std::tuple<int, int, int, int> a, std::tuple<int, int, int, int> b);
