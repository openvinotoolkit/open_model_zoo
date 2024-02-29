// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/args_helper.hpp"
#include "utils_gapi/backend_description.hpp"
#include "utils_gapi/kernel_package.hpp"

BackendDescription BackendDescription::parseFromArgs(const std::string &arg, char sep) {
    std::vector<std::string> splitted_line = split(arg, sep);
    if (splitted_line.empty()) {
        throw std::runtime_error("Cannot parse BackendDescription from string: " + arg +
                                 ". Backend name and its fields must be separated by \"" + sep  +
                                 "\". Example: <backend>" + sep + "<provider> " + sep + "<device>");
    }

    auto props_it = splitted_line.begin();
    std::advance(props_it, 1);
    return BackendDescription(splitted_line[0], props_it, splitted_line.end());
}

BackendsConfig::BackendsConfig(const ModelConfig &src,
                               const std::vector<float> &mean_values,
                               const std::vector<float> &scale_values) :
    ModelConfig(src),
    mean_values(mean_values),
    scale_values(scale_values) {
}

std::initializer_list<std::string> getSupportedInferenceBackends() {
    static const std::initializer_list<std::string> backends{
#ifdef GAPI_IE_BACKEND
        "IE"
#endif // GAPI_IE_BACKEND
#ifdef GAPI_ONNX_BACKEND
        , "ONNX"
#endif // GAPI_ONNX_BACKEND
#ifdef GAPI_OV_BACKEND
        , "OV"
#endif //GAPI_OV_BACKEND
    };
    return backends;
}
