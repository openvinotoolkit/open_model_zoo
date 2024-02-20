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
            
        if (!config.mean_values.empty() && !config.scale_values.empty()) {
            std::vector<float> means;
            split(config.mean_values, ' ', means);

            std::vector<float> scales;
            split(config.scale_values, ' ', scales);

            if (means.size() != 3  || scales.size() != 3) {
                throw std::runtime_error("`mean_values` and `scale_values` must be 3-components vectors "
                                         "with a space symbol as separator between component values");
            }
            net.cfgMean(means).cfgScale(scales);
        }
        else if (!config.mean_values.empty() && config.scale_values.empty() || config.mean_values.empty() && !config.scale_values.empty() ) {
            throw std::runtime_error(backends.front().name + " requires both `mean_values` and 'scale_values' "
                                     "to be set or both unset. Please check the command arguments");
        }

        return cv::gapi::networks(net);
    }
};

#endif // GAPI_OV_BACKEND
