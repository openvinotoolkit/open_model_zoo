// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <list>
#include <map>
#include <stdexcept>
#include <string>

#include "ie_backend.hpp"
#include "onnx_backend.hpp"
#include "ov_backend.hpp"

template<class ExecNetwork>
cv::gapi::GNetPackage create_execution_network(const std::string &model_path,
                                               const BackendsConfig &config,
                                               const inference_backends_t &backends = inference_backends_t{}) {
    if (backends.empty()) {
        throw std::runtime_error("No G-API backend specified.\nPlease select a backend from the list: " +
                                 merge(getSupportedInferenceBackends(), ", "));
    }
    static const std::map<std::string,
                          std::function<cv::gapi::GNetPackage(const std::string &,
                                                              const BackendsConfig &,
                                                              const inference_backends_t &)>
                         > maps {
#ifdef GAPI_IE_BACKEND
        {"IE",      &BackendApplicator<ExecNetwork, cv::gapi::ie::Params>::apply}
#endif
#ifdef GAPI_ONNX_BACKEND
        , { "ONNX", &BackendApplicator<ExecNetwork, cv::gapi::onnx::Params>::apply}
#endif
#ifdef GAPI_OV_BACKEND
        , { "OV",   &BackendApplicator<ExecNetwork, cv::gapi::ov::Params>::apply}
#endif
    };

    const BackendDescription &backend = backends.front();
    const auto it = maps.find(backend.name);
    if (it == maps.end()) {
        throw std::runtime_error("Cannot apply unknown G-API backend: " + backend.name +
                                 "\nPlease, check on available backend list: " +
                                 merge(getSupportedInferenceBackends(), ","));
    }
    return it->second(model_path, config, backends);
}
