// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <queue>
#include <string>
#include <vector>

#include "utils_gapi/gapi_features.hpp"

struct BackendDescription {
    std::string name;

    std::vector<std::string> properties;
    static BackendDescription parseFromArgs(const std::string &arg, char sep = '/');
private:
    BackendDescription(const std::vector<std::string> &fields);
};

using inference_backends_t = std::queue<BackendDescription>;

std::initializer_list<std::string> getSupportedInferenceBackends();
