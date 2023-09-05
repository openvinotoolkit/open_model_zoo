// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/infer/ov.hpp>

#include <utils/args_helper.hpp>

#include "backend_description.hpp"
#include "onnx_execution_provider.hpp"

template<class ExecNetwork>
using backend_applicator_t = std::function<cv::gapi::GNetPackage(const std::string &,
                                                                 const ModelConfig &,
                                                                 const inference_backends_t &)>;
template <class ExecNetwork>
static cv::gapi::GNetPackage applyIEBackend(const std::string &model_path,
                                            const ModelConfig &config,
                                            const inference_backends_t &backends) {
    const BackendDescription &backend = backends.front();
    if(!backend.properties.empty()) {
        throw std::runtime_error("IE backend doesn't support any arguments");
    }
    const auto net =
            cv::gapi::ie::Params<ExecNetwork>{
                model_path,                          // path to topology IR
                fileNameNoExt(model_path) + ".bin",  // path to weights
                config.deviceName                    // device specifier
            }.cfgNumRequests(config.maxAsyncRequests)
            .pluginConfig(config.getLegacyConfig());
    return cv::gapi::networks(net);
}

template <class ExecNetwork>
static cv::gapi::GNetPackage applyOVBackend(const std::string &model_path,
                                            const ModelConfig &config,
                                            const inference_backends_t &backends) {
    const BackendDescription &backend = backends.front();
    if(!backend.properties.empty()) {
        throw std::runtime_error("OV backend doesn't support any arguments");
    }
    const auto net =
            cv::gapi::ov::Params<ExecNetwork>{
                model_path,                          // path to topology IR
                fileNameNoExt(model_path) + ".bin",  // path to weights
                config.deviceName                    // device specifier
            }.cfgNumRequests(config.maxAsyncRequests)
            .cfgPluginConfig(config.getLegacyConfig());
    return cv::gapi::networks(net);
}

template <class ExecNetwork>
static cv::gapi::GNetPackage applyONNXBackend(const std::string &model_path,
                                              const ModelConfig &config,
                                              const inference_backends_t &backends) {
    auto net =
            cv::gapi::onnx::Params<ExecNetwork>{
                model_path
            };
    applyONNXProviders(net, config, backends);
    return cv::gapi::networks(net);
}


template<class ExecNetwork>
cv::gapi::GNetPackage applyBackend(const std::string &model_path,
                                   const ModelConfig &config,
                                   inference_backends_t backends = inference_backends_t{}) {
    if (backends.empty()) {
       throw std::runtime_error("No G-API backend specified");
    }
    static const std::map<std::string, backend_applicator_t<ExecNetwork>> maps{
#ifdef GAPI_IE_BACKEND
        {"IE",  &applyIEBackend<ExecNetwork>}
#endif
#ifdef GAPI_ONNX_BACKEND
        , { "ONNX", &applyONNXBackend<ExecNetwork>}
#endif
#ifdef GAPI_OV_BACKEND
        , { "OV", applyOVBackend<ExecNetwork>}
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
