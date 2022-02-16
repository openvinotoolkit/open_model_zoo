// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

cv::Scalar getNetShape(const std::string& path);

void erase(std::string& str, const char symbol);

template<typename... Sargs>
void erase(std::string& str, const char symbol, Sargs... symbols);

std::vector<std::string> fill_labels(const std::string& dir);
