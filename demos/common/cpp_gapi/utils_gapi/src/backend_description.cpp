// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/args_helper.hpp"
#include "utils_gapi/backend_description.hpp"
#include "utils_gapi/kernel_package.hpp"

BackendDescription::BackendDescription(const std::vector<std::string> &fields) {
    name = fields.at(0);

    auto fields_it = fields.begin();
    std::advance(fields_it, 1);
    properties.insert(properties.end(), fields_it, fields.end());
}

BackendDescription BackendDescription::parseFromArgs(const std::string &arg, char sep) {
    std::vector<std::string> splitted_line = split(arg, sep);
    if (splitted_line.empty()) {
        throw std::runtime_error("Cannot parse BackendDescription from string: " + arg +
                                 ". Backend name and its fields must be separated by \"" + sep  +
                                 "\". Example: <backend>" + sep + "<provider> " + sep + "<device>");
    }
    return BackendDescription(splitted_line);
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
