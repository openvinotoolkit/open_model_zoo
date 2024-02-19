// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "cnn.hpp"

std::string CTCGreedyDecoder(const std::vector<float>& data, const std::string& alphabet, char pad_symbol, double* conf);
std::string CTCBeamSearchDecoder(const std::vector<float>& data, const std::string& alphabet, char pad_symbol, double* conf, int bandwidth);
std::string SimpleDecoder(const std::vector<float>& data, const std::string& alphabet, char pad_symbol, double* conf, int start_index);

class TextRecognizer : public Cnn {
public:
    TextRecognizer(
        const std::string& model_path, const std::string& model_type, const std::string& deviceName,
        ov::Core& core,
        const std::string& out_enc_hidden_name,
        const std::string& out_dec_hidden_name,
        const std::string& in_dec_hidden_name,
        const std::string& features_name,
        const std::string& in_dec_symbol_name,
        const std::string& out_dec_symbol_name,
        const std::string& logits_name,
        size_t end_token,
        bool use_auto_resize = false);

    std::map<std::string, ov::Tensor> Infer(const cv::Mat& frame) override;

    const cv::Size& input_size() const { return m_input_size; }

private:
    void check_model_names(
        const ov::OutputVector& input_info_decoder, const ov::OutputVector& output_info_decoder) const;

    bool m_isCompositeModel;
    ov::InferRequest m_infer_request_decoder;
    std::string m_features_name;
    std::string m_out_enc_hidden_name;
    std::string m_out_dec_hidden_name;
    std::string m_in_dec_hidden_name;
    std::string m_in_dec_symbol_name;
    std::string m_out_dec_symbol_name;
    std::string m_logits_name;
    size_t m_end_token;
};
