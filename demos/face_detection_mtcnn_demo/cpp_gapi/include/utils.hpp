// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi.hpp>

//Infer helper function
namespace {
static inline std::tuple<cv::GMat, cv::GMat> run_mtcnn_p(cv::GMat &in, const std::string &id) {
    cv::GInferInputs inputs;
    inputs["data"] = in;
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(id, inputs);
    auto regressions = outputs.at("conv4-2");
    auto scores = outputs.at("prob1");
    return std::make_tuple(regressions, scores);
}

static inline std::string get_pnet_level_name(const cv::Size &in_size) {
    return "MTCNNProposal_" + std::to_string(in_size.width) + "x" + std::to_string(in_size.height);
}

int calculate_scales(const cv::Size &input_size, std::vector<double> &out_scales, std::vector<cv::Size> &out_sizes ) {
    //calculate multi - scale and limit the maxinum side to 1000
    //pr_scale: limit the maxinum side to 1000, < 1.0
    double pr_scale = 1.0;
    double h = static_cast<double>(input_size.height);
    double w = static_cast<double>(input_size.width);
    if (std::min(w, h) > 1000)
    {
        pr_scale = 1000.0 / std::min(h, w);
        w = w * pr_scale;
        h = h * pr_scale;
    }
    else if (std::max(w, h) < 1000)
    {
        w = w * pr_scale;
        h = h * pr_scale;
    }
    //multi - scale
    out_scales.clear();
    out_sizes.clear();
    const double factor = 0.709;
    int factor_count = 0;
    double minl = std::min(h, w);
    while (minl >= 12)
    {
        const double current_scale = pr_scale * std::pow(factor, factor_count);
        cv::Size current_size(static_cast<int>(static_cast<double>(input_size.width) * current_scale),
                              static_cast<int>(static_cast<double>(input_size.height) * current_scale));
        out_scales.push_back(current_scale);
        out_sizes.push_back(current_size);
        minl *= factor;
        factor_count += 1;
    }
    return factor_count;
}

int calculate_half_scales(const cv::Size &input_size, std::vector<double>& out_scales, std::vector<cv::Size>& out_sizes) {
    double pr_scale = 0.5;
    const double h = static_cast<double>(input_size.height);
    const double w = static_cast<double>(input_size.width);
    //multi - scale
    out_scales.clear();
    out_sizes.clear();
    const double factor = 0.5;
    int factor_count = 0;
    double minl = std::min(h, w);
    while (minl >= 12.0*2.0)
    {
        const double current_scale = pr_scale;
        cv::Size current_size(static_cast<int>(static_cast<double>(input_size.width) * current_scale),
                              static_cast<int>(static_cast<double>(input_size.height) * current_scale));
        out_scales.push_back(current_scale);
        out_sizes.push_back(current_size);
        minl *= factor;
        factor_count += 1;
        pr_scale *= 0.5;
    }
    return factor_count;
}

} // anonymous namespace
