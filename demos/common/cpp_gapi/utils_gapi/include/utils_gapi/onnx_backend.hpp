// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <list>
#include <map>
#include <stdexcept>
#include <string>

#include "utils_gapi/backend_description.hpp"

#ifdef GAPI_ONNX_BACKEND
#include <opencv2/gapi/infer/onnx.hpp>

const std::initializer_list<std::string> getONNXSupportedEP();

template<class ExecNetwork, class ...Args>
void applyONNXProviders(ExecNetwork& net, const ModelConfig &config, inference_backends_t backend_cfgs = inference_backends_t{});

template<class ExecNetwork>
struct BackendApplicator<ExecNetwork,
                         cv::gapi::onnx::Params> {
    static cv::gapi::GNetPackage apply(const std::string &model_path, const BackendsConfig &config, const inference_backends_t &backends) {
        auto net =
            cv::gapi::onnx::Params<ExecNetwork>{
                model_path
            };

        if (config.mean_values.empty() && config.scale_values.empty()) {
            net.cfgNormalize({false});
        } else if (!config.mean_values.empty() && !config.scale_values.empty()) {
            net.cfgMeanStd({cv::Scalar(config.mean_values[0], config.mean_values[1], config.mean_values[2])},
                           {cv::Scalar(config.scale_values[0], config.scale_values[1], config.scale_values[2])});
        } else {
            throw std::runtime_error(backends.front().name + " requires both `mean_values` and 'scale_values' "
                                     "to be set or both unset. Please check the command arguments");
        }

        applyONNXProviders(net, config, backends);
        return cv::gapi::networks(net);
    }
};

namespace {
struct CPUProvider{};
template<class ExecNetwork, class Provider = CPUProvider>
struct ProviderApplicator {
    static void apply(ExecNetwork &, const BackendsConfig &, const inference_backends_t &backends) {
        static_assert(std::is_same<Provider, CPUProvider>::value, "Unsupported ONNX provider requested. Please add a partial specialization if required");
        if (backends.size() > 1) {
            throw std::runtime_error("ONNX CPU execution provider must be only provider in the provider list or "
                                     "be the last in order");
        }

        const BackendDescription &backend = backends.front();
        if (backend.properties.size() >= 2) {
            throw std::runtime_error("ONNX CPU execution provider doesn't support any arguments");
        }
    }
};

#ifdef GAPI_ONNX_BACKEND_EP_EXTENSION
template<class ExecNetwork>
struct ProviderApplicator<ExecNetwork, cv::gapi::onnx::ep::DirectML> {
    static void apply(ExecNetwork &net, const BackendsConfig &config, const inference_backends_t &backends) {
        std::string ep_selected_device = config.deviceName;
        const BackendDescription &backend = backends.front();
        if (backend.properties.size() >= 2) {
            ep_selected_device = backend.properties[1]; // second arg for ONNX is device
        }
        net.cfgAddExecutionProvider(cv::gapi::onnx::ep::DirectML(ep_selected_device))
           .cfgDisableMemPattern();
    }
};

template<class ExecNetwork>
struct ProviderApplicator<ExecNetwork, cv::gapi::onnx::ep::OpenVINO> {
    static void apply(ExecNetwork &net, const BackendsConfig &config, const inference_backends_t &backends) {
        std::string ep_selected_device = config.deviceName;
        const BackendDescription &backend = backends.front();
        if (backend.properties.size() >= 2) {
            ep_selected_device = backend.properties[1]; // second arg for ONNX is device
        }
        net.cfgAddExecutionProvider(cv::gapi::onnx::ep::OpenVINO(ep_selected_device));
    }
};
#endif // GAPI_ONNX_BACKEND_EP_EXTENSION
}

template<class ExecNetwork, class ...Args>
void applyONNXProviders(ExecNetwork& net, const BackendsConfig &config, inference_backends_t backend_cfgs) {
    if (backend_cfgs.empty()) {
        return;
    }
    static const std::map<std::string,
                          std::function<void(ExecNetwork&,
                                             const BackendsConfig &,
                                             const inference_backends_t &)>
                         > maps {
        {"CPU",  &ProviderApplicator<ExecNetwork>::apply},
        {"",     &ProviderApplicator<ExecNetwork>::apply},
#ifdef GAPI_ONNX_BACKEND_EP_EXTENSION
        {"DML",  &ProviderApplicator<ExecNetwork, cv::gapi::onnx::ep::DirectML>::apply},
        {"OV",   &ProviderApplicator<ExecNetwork, cv::gapi::onnx::ep::OpenVINO>::apply}
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
            ProviderApplicator<ExecNetwork, CPUProvider>::apply(net, config, backend_cfgs);
        }
        // pop backed/provider processed
        backend_cfgs.pop();
    }
}
#endif // GAPI_ONNX_BACKEND
