// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

cv::Scalar getNetShape(const std::string& path);

void erase(std::string& str, const char symbol);

template <typename... Sargs>
void erase(std::string& str, const char symbol, Sargs... symbols);

std::vector<std::string> fill_labels(const std::string& dir);
