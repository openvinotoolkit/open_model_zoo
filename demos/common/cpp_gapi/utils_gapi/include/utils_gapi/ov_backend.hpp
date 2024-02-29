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

#ifdef GAPI_OV_BACKEND
#include <opencv2/gapi/infer/ov.hpp>

template<class ExecNetwork>
struct BackendApplicator<ExecNetwork,
                         cv::gapi::ov::Params> {
    static cv::gapi::GNetPackage apply(const std::string &model_path, const BackendsConfig &config, const inference_backends_t &backends) {
        const BackendDescription &backend = backends.front();
        if(!backend.properties.empty()) {
            throw std::runtime_error(backend.name + " backend doesn't support any arguments. Please remove: " +
                                     merge(backend.properties, "/") + " from backend argument list");
        }
        auto net =
            cv::gapi::ov::Params<ExecNetwork>{
                model_path,                          // path to topology IR
                fileNameNoExt(model_path) + ".bin",  // path to weights
                config.deviceName                    // device specifier
            }.cfgNumRequests(config.maxAsyncRequests)
            .cfgPluginConfig(config.getLegacyConfig());

        if(!config.mean_values.empty()){
            net.cfgMean(config.mean_values);
        }
        if(!config.scale_values.empty()) {
            net.cfgScale(config.scale_values);
        }
        return cv::gapi::networks(net);
    }
};

#endif // GAPI_OV_BACKEND
