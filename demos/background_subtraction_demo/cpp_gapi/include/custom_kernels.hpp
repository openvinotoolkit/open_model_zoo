// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>  // for string
#include <vector>  // for vector

#include <cpp/ie_cnn_network.h>  // for CNNNetwork
#include <ie_allocator.hpp>  // for InferenceEngine
#include <ie_common.h>  // for OutputsDataMap
#include <ie_input_info.hpp>  // for InputsDataMap
#include <opencv2/core.hpp>  // for Size, CV_8U
#include <opencv2/gapi/gkernel.hpp>  // for G_API_OP, KernelTypeMedium, GKernelPackage
#include <opencv2/gapi/gmat.hpp>  // for GMatDesc, GMat

namespace IE = InferenceEngine;

namespace custom {
// clang-format off
G_API_OP(GTensorToImg, <cv::GMat(cv::GMat)>, "custom.tensorToImg") {
    static cv::GMatDesc outMeta(const cv::GMatDesc& in) {
        // NB: Input is ND mat.
        return cv::GMatDesc{in.depth, in.dims[1], cv::Size(in.dims[3], in.dims[2])};
    }
};

G_API_OP(GCalculateMaskRCNNBGMask,
         <cv::GMat(cv::Size, cv::Size, cv::GMat, cv::GMat, cv::GMat)>,
         "maskrcnn.calculate-background-mask") {
    static cv::GMatDesc outMeta(const cv::Size& in_sz,
                                const cv::Size&,
                                const cv::GMatDesc&,
                                const cv::GMatDesc&,
                                const cv::GMatDesc&) {
        return cv::GMatDesc{CV_8U, 1, in_sz};
    }
};
// clang-format on
class NNBGReplacer {
public:
    NNBGReplacer() = default;
    virtual ~NNBGReplacer() = default;
    NNBGReplacer(const std::string& model_path);
    virtual cv::GMat replace(cv::GMat, const cv::Size&, cv::GMat) = 0;
    const std::string& getName() {
        return m_tag;
    }

protected:
    IE::CNNNetwork m_cnn_network;
    std::string m_tag;
    IE::InputsDataMap m_inputs;
    IE::OutputsDataMap m_outputs;
};

class MaskRCNNBGReplacer : public NNBGReplacer {
public:
    MaskRCNNBGReplacer(const std::string& model_path);
    cv::GMat replace(cv::GMat, const cv::Size&, cv::GMat) override;

private:
    std::string m_input_name;
    std::string m_labels_name;
    std::string m_boxes_name;
    std::string m_masks_name;
};

class BGMattingReplacer : public NNBGReplacer {
public:
    BGMattingReplacer(const std::string& model_path);
    cv::GMat replace(cv::GMat, const cv::Size&, cv::GMat) override;

private:
    std::string m_input_name;
    std::string m_output_name;
};

cv::gapi::GKernelPackage kernels();

}  // namespace custom
