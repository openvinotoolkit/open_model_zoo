// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <queue>
#include <string>
#include <vector>

#include <opencv2/gapi/infer.hpp>
#include <utils/config_factory.h>

#include "utils_gapi/gapi_features.hpp"

struct BackendDescription {
    template<class It>
    BackendDescription(const std::string &name, It begin, It end) :
        name(name), properties(begin, end) {}

    static BackendDescription parseFromArgs(const std::string &arg, char sep = '/');

    std::string name;
    std::vector<std::string> properties;
};

struct BackendsConfig: ModelConfig {
    BackendsConfig(const ModelConfig &src,
                   const std::string &mean_values = "",
                   const std::string &scale_values = "");
    std::string mean_values;
    std::string scale_values;
};

using inference_backends_t = std::queue<BackendDescription>;

std::initializer_list<std::string> getSupportedInferenceBackends();

template<class ExecNetwork,
         template <class> class Params>
struct BackendApplicator {
    static cv::gapi::GNetPackage apply(const std::string&, const BackendsConfig &, const inference_backends_t &);
};
