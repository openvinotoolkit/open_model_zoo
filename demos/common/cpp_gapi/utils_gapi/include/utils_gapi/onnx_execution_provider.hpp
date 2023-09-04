// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <list>
#include <map>
#include <stack>
#include <stdexcept>
#include <string>

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/onnx.hpp>

#include <utils/args_helper.hpp>
#include <utils/config_factory.h>

#include "utils_gapi/backend_description.hpp"


const std::initializer_list<std::string> getONNXSupportedEP();

namespace {
template<class ExecNetwork>
using provider_applicator_t = std::function<void(ExecNetwork&,
                                                 const ModelConfig &,
                                                 const inference_backends_t &)>;

template <class ExecNetwork>
static void applyDefaultProvider(ExecNetwork &,
                                 const ModelConfig &,
                                 const inference_backends_t &backends) {
    if (backends.size() > 1) {
        throw std::runtime_error("ONNX CPU execution provider must be only provider in the provider list or "
                                 "be the last in order");
    }
}

template <class ExecNetwork>
static void applyDMLProvider(ExecNetwork &net, const ModelConfig &config, const inference_backends_t &backends) {
    std::string ep_selected_device = config.deviceName;

    const BackendDescription &backend = backends.front();
    if(backend.properties.size() >= 2) {
        ep_selected_device = backend.properties[1]; // second arg for ONNX is device
    }
    net.cfgAddExecutionProvider(cv::gapi::onnx::ep::DirectML(ep_selected_device))
       .cfgDisableMemPattern();
}

template <class ExecNetwork>
static void applyOVEPProvider(ExecNetwork &net, const ModelConfig &config, const inference_backends_t &backends) {
    std::string ep_selected_device = config.deviceName;
    const BackendDescription &backend = backends.front();
    if(backend.properties.size() >= 2) {
        ep_selected_device = backend.properties[1]; // second arg for ONNX is device
    }
    net.cfgAddExecutionProvider(cv::gapi::onnx::ep::OpenVINO(ep_selected_device));
}
}

template<class ExecNetwork, class ...Args>
void applyONNXProviders(ExecNetwork& net, const ModelConfig &config, inference_backends_t backend_cfgs = inference_backends_t{}) {
    if (backend_cfgs.empty()) {
        return;
    }
    static const std::map<std::string, provider_applicator_t<ExecNetwork>> maps{
        {"CPU",  &applyDefaultProvider<ExecNetwork>},
        {"",     &applyDefaultProvider<ExecNetwork>},
#ifdef GAPI_ONNX_BACKEND_EP_EXTENSION
        {"DML",  &applyDMLProvider<ExecNetwork>},
        {"OVEP", &applyOVEPProvider<ExecNetwork>}
#endif // GAPI_ONNX_BACKEND_EP_EXTENSION
    };

    while(!backend_cfgs.empty()) {
        // find & apply ONNX execution providers
        const BackendDescription &backend = backend_cfgs.front();
        if (backend.properties.size() >= 1) {
            // the first of ONNX backend is execution provider
            const std::string &execution_provider = backend.properties[0];
            const auto it = maps.find(execution_provider);
            if (it == maps.end()) {
                throw std::runtime_error("Cannot apply unknown provider: \"" + execution_provider +
                                        "\" for ONNX backend\nPlease, check on the available ONNX providers list: " +
                                        merge(getONNXSupportedEP(), ", "));
            }
            it->second(net, config, backend_cfgs);
        } else {
            applyDefaultProvider(net, config, backend_cfgs);
        }
        // pop backed/provider processed
        backend_cfgs.pop();
    }
}

