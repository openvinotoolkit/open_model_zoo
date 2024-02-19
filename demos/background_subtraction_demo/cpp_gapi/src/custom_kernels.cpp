// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_kernels.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

#include <ie_core.hpp>
#include <ie_data.h>
#include <ie_layouts.h>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/operators.hpp>
#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/imgproc.hpp>

// clang-format off
GAPI_OCV_KERNEL(OCVTensorToImg, custom::GTensorToImg) {
    static void run(const cv::Mat &in,
                    cv::Mat &out) {
        auto* out_p = out.ptr<float>();
        auto* in_p  = in.ptr<const float>();
        std::copy_n(in_p, in.size[2] * in.size[3], out_p);
    }
};
// clang-format on

static cv::Rect expandBox(const float x0, const float y0, const float x1, const float y1, const float scale) {
    const float w_half = ((x1 - x0) * 0.5f) * scale;
    const float h_half = ((y1 - y0) * 0.5f) * scale;
    const float x_c = (x1 + x0) * 0.5f;
    const float y_c = (y1 + y0) * 0.5f;
    cv::Point tl(static_cast<int>(x_c - w_half), static_cast<int>(y_c - h_half));
    cv::Point br(static_cast<int>(x_c + w_half), static_cast<int>(y_c + h_half));
    return cv::Rect(tl, br);
}

static int clip(int value, int lower, int upper) {
    return std::max(lower, std::min(value, upper));
}

// clang-format off
GAPI_OCV_KERNEL(OCVCalculateMaskRCNNBGMask, custom::GCalculateMaskRCNNBGMask) {
    static void run(const cv::Size& original_sz,
                    const cv::Size& model_in_size,
                    const cv::Mat& raw_labels,
                    const cv::Mat& raw_boxes,
                    const cv::Mat& raw_masks,
                    cv::Mat& mask) {
        GAPI_Assert(raw_labels.depth() == CV_32S);
        GAPI_Assert(raw_boxes.depth()  == CV_32F);
        GAPI_Assert(raw_masks.depth()  == CV_32F);

        const int BOX_SIZE = 5;
        const int CONFIDENCE_OFFSET = 4;
        const int BOX_X0_OFFSET = 0;
        const int BOX_Y0_OFFSET = 1;
        const int BOX_X1_OFFSET = 2;
        const int BOX_Y1_OFFSET = 3;
        const int PERSON_ID = 0;
        const float BIN_THRESHOLD = 0.5f;

        const int num_boxes = raw_boxes.size[0];
        const float* boxes_ptr = raw_boxes.ptr<float>();
        // FIXME: In order to create "view" over raw_masks, need to obtain non const pointer.
        float* masks_ptr = const_cast<float*>(raw_masks.ptr<float>());
        const int* labels_ptr = raw_labels.ptr<int>();
        const float prob_threshold = 0.5;
        const cv::Size mask_sz(raw_masks.size[1], raw_masks.size[2]);

        const float scale_x = static_cast<float>(model_in_size.width)  / original_sz.width;
        const float scale_y = static_cast<float>(model_in_size.height) / original_sz.height;
        const float scale = 1.f;

        std::vector<cv::Mat> person_masks;
        for (int box_idx = 0; box_idx < num_boxes; ++box_idx) {
            const int conf_idx = box_idx * BOX_SIZE + CONFIDENCE_OFFSET;

            if (boxes_ptr[conf_idx] > prob_threshold &&
                labels_ptr[box_idx] == PERSON_ID) {
                const int mask_offset     = box_idx * mask_sz.area();
                const int box_pos_offset  = box_idx * BOX_SIZE;

                auto box = expandBox(boxes_ptr[box_pos_offset + BOX_X0_OFFSET] / scale_x,
                                     boxes_ptr[box_pos_offset + BOX_Y0_OFFSET] / scale_y,
                                     boxes_ptr[box_pos_offset + BOX_X1_OFFSET] / scale_x,
                                     boxes_ptr[box_pos_offset + BOX_Y1_OFFSET] / scale_y,
                                     scale);

                const auto& br  = box.br();
                const auto& tl  = box.tl();

                const int x0 = clip(tl.x, 0, original_sz.width);
                const int y0 = clip(tl.y, 0, original_sz.height);
                const int x1 = clip(br.x, 0, original_sz.width);
                const int y1 = clip(br.y, 0, original_sz.height);

                cv::Mat person_mask(mask_sz, CV_32FC1, masks_ptr + mask_offset);
                cv::Mat resized_person_mask;

                const auto diff = br - tl;
                const cv::Size expanded_size(std::max(diff.x + 1, 1),
                                             std::max(diff.y + 1, 1));
                cv::resize(person_mask, resized_person_mask, expanded_size);

                cv::Mat bin_mask = resized_person_mask > BIN_THRESHOLD;
                bin_mask.convertTo(bin_mask, CV_8U, 255);

                auto bin_mask_slice = cv::Rect(cv::Point(x0 - tl.x, y0 - tl.y),
                                               cv::Point(x1 - tl.x, y1 - tl.y));

                cv::Mat result_mask(original_sz, CV_8UC1, cv::Scalar(0));
                bin_mask(bin_mask_slice).copyTo(result_mask(cv::Rect(cv::Point(x0, y0),
                                                                     cv::Point(x1, y1))));
                person_masks.push_back(result_mask);
            }
        }

        if (person_masks.empty()) {
            mask.setTo(255.f);
        } else {
            person_masks[0].copyTo(mask);
            for (const auto& m : person_masks) {
                mask |= m;
            }
        }
    }
};
// clang-format on

custom::NNBGReplacer::NNBGReplacer(const std::string& model_path) {
    IE::Core core;
    m_cnn_network = core.ReadNetwork(model_path);
    m_tag = m_cnn_network.getName();
    m_inputs = m_cnn_network.getInputsInfo();
    m_outputs = m_cnn_network.getOutputsInfo();
}

custom::MaskRCNNBGReplacer::MaskRCNNBGReplacer(const std::string& model_path) : custom::NNBGReplacer(model_path) {
    for (const auto& p : m_outputs) {
        const auto& layer_name = p.first;
        if (layer_name.rfind("TopK") != std::string::npos) {
            continue;
        }

        if (m_inputs.size() != 1) {
            throw std::logic_error("Supported only single input MaskRCNN models!");
        }
        m_input_name = m_inputs.begin()->first;

        const auto dims_size = p.second->getTensorDesc().getDims().size();
        if (dims_size == 1) {
            m_labels_name = layer_name;
        } else if (dims_size == 2) {
            m_boxes_name = layer_name;
        } else if (dims_size == 3) {
            m_masks_name = layer_name;
        } else {
            throw std::logic_error("Unexpected output layer shape for: " + layer_name);
        }
    }
}

cv::GMat custom::MaskRCNNBGReplacer::replace(cv::GFrame in, cv::GMat bgr, const cv::Size& in_size, cv::GMat background) {
    cv::GInferInputs inputs;
    inputs[m_input_name] = in;
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(m_tag, inputs);
    auto labels = outputs.at(m_labels_name);
    auto boxes = outputs.at(m_boxes_name);
    auto masks = outputs.at(m_masks_name);

    const auto& dims = m_inputs.at(m_input_name)->getTensorDesc().getDims();
    GAPI_Assert(dims.size() == 4u);
    auto mask = custom::GCalculateMaskRCNNBGMask::on(in_size, cv::Size(dims[3], dims[2]), labels, boxes, masks);
    auto mask3ch = cv::gapi::medianBlur(cv::gapi::merge3(mask, mask, mask), 11);
    return (mask3ch & bgr) + (~mask3ch & background);
}

custom::BGMattingReplacer::BGMattingReplacer(const std::string& model_path) : NNBGReplacer(model_path) {
    if (m_inputs.size() != 1) {
        throw std::logic_error("Supported only single input background matting models!");
    }
    m_input_name = m_inputs.begin()->first;

    if (m_outputs.size() != 1) {
        throw std::logic_error("Supported only single output background matting models!");
    }
    m_output_name = m_outputs.begin()->first;
}

cv::GMat custom::BGMattingReplacer::replace(cv::GFrame in, cv::GMat bgr, const cv::Size& in_size, cv::GMat background) {
    cv::GInferInputs inputs;
    inputs[m_input_name] = in;
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(m_tag, inputs);

    auto alpha = cv::gapi::resize(custom::GTensorToImg::on(outputs.at(m_output_name)), in_size);
    auto alpha3ch = cv::gapi::merge3(alpha, alpha, alpha);
    auto in_fp = cv::gapi::convertTo(bgr, CV_32F);
    auto bgr_fp = cv::gapi::convertTo(background, CV_32F);

    cv::GScalar one(cv::Scalar::all(1.));
    auto out = cv::gapi::mul(alpha3ch, in_fp) + cv::gapi::mul((one - alpha3ch), bgr_fp);
    return cv::gapi::convertTo(out, CV_8U);
}

cv::gapi::GKernelPackage custom::kernels() {
    return cv::gapi::kernels<OCVTensorToImg, OCVCalculateMaskRCNNBGMask>();
}
