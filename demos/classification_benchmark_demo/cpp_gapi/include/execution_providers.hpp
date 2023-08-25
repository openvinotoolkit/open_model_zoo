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
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/infer/onnx.hpp>

using execution_provider_t = std::string;
using execution_provider_arg_t = std::string;
using execution_provider_desription_t = std::pair<execution_provider_t, execution_provider_arg_t>;
using execution_providers_t = std::stack<execution_provider_desription_t>;


inline const std::list<execution_provider_t> getSupportedEP() {
#ifdef GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
    static const std::list<execution_provider_t> providers{"DML", "OVEP"};
    return providers;
#else // GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
    return {};
#endif // GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
}

execution_providers_t createProvidersFromString(const std::string &str);

#ifdef GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
namespace {
template<class ExecNetwork>
using provider_applicator_t = std::function<void(ExecNetwork&, const std::string&)>;

template <class ExecNetwork>
static void applyDMLProvider(ExecNetwork &net, const std::string &device_name) {
    net.cfgAddExecutionProvider(cv::gapi::onnx::ep::DirectML(device_name))
       .cfgDisableMemPattern();
}

template <class ExecNetwork>
static void applyOVEPProvider(ExecNetwork &net, const std::string &device_name) {
    net.cfgAddExecutionProvider(cv::gapi::onnx::ep::OpenVINO(device_name));
}
}

template<class ExecNetwork, class ...Args>
void applyProvider(ExecNetwork& net, const execution_provider_desription_t &multiple_ep = execution_provider_desription_t{})
    static const std::map<execution_provider_t, provider_applicator_t<ExecNetwork>> maps{
        {"DML",  &applyDMLProvider<ExecNetwork>},
        {"OVEP", &applyOVEPProvider<ExecNetwork>}};
    if (multiple_ep.empty()) {
        return net;
    }

    execution_provider_t provider;
    execution_provider_device_t device;
    while(!multiple_ep.empty()) {
        std::tie(provider, device) = multiple_ep.top();
        multiple_ep.pop();

        const auto it = maps.find(provider);
        if (it == maps.end()) {
            throw std::runtime_error("Cannot apply unknown provider: " + provider);
        }

        it->second(net, device);
    }
}
#endif // GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
