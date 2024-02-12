// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <list>
#include <map>
#include <stdexcept>
#include <string>

#include <utils/args_helper.hpp>
#include "utils_gapi/backend_description.hpp"

#ifdef GAPI_IE_BACKEND
#include <opencv2/gapi/infer/ie.hpp>

template<class ExecNetwork>
struct BackendApplicator<ExecNetwork,
                         cv::gapi::ie::Params> {
    static cv::gapi::GNetPackage apply(const std::string &model_path, const BackendsConfig &config, const inference_backends_t &backends) {
        const BackendDescription &backend = backends.front();
        if(!backend.properties.empty()) {
            throw std::runtime_error(backend.name + " backend doesn't support any arguments. Please remove: " +
                                     merge(backend.properties, "/") + " from backend argument list");
        }

        if (!config.mean_values.empty() || !config.scale_values.empty()) {
            throw std::runtime_error(backend.name + " backend doesn't support neither 'mean_values' nor 'scale_values`");
        }

        auto net =
            cv::gapi::ie::Params<ExecNetwork>{
                model_path,                          // path to topology IR
                fileNameNoExt(model_path) + ".bin",  // path to weights
                config.deviceName                    // device specifier
            }.cfgNumRequests(config.maxAsyncRequests)
            .pluginConfig(config.getLegacyConfig());
        return cv::gapi::networks(net);
    }
};
#endif // GAPI_IE_BACKEND
