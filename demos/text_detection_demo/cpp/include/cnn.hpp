// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <utils/ocv_common.hpp>

using namespace InferenceEngine;

class Cnn {
  public:
    Cnn(const std::string &model_path, Core & ie, const std::string & deviceName,
              const cv::Size &new_input_resolution = cv::Size());

    virtual InferenceEngine::BlobMap Infer(const cv::Mat &frame);

    size_t ncalls() const {return ncalls_;}
    double time_elapsed() const {return time_elapsed_;}
    const cv::Size& input_size() const {return input_size_;}

  protected:
    cv::Size input_size_;
    int channels_;
    std::string input_name_;
    InferRequest infer_request_;
    std::vector<std::string> output_names_;

    double time_elapsed_;
    size_t ncalls_;
};

class EncoderDecoderCNN : public Cnn {
  public:
    EncoderDecoderCNN(std::string model_path,
                      Core &ie, const std::string &deviceName,
                      const std::string &out_enc_hidden_name,
                      const std::string &out_dec_hidden_name,
                      const std::string &in_dec_hidden_name,
                      const std::string &features_name,
                      const std::string &in_dec_symbol_name,
                      const std::string &out_dec_symbol_name,
                      const std::string &logits_name,
                      size_t end_token
                      );
    InferenceEngine::BlobMap Infer(const cv::Mat &frame) override;
  private:
    InferRequest infer_request_decoder_;
    std::string features_name_;
    std::string out_enc_hidden_name_;
    std::string out_dec_hidden_name_;
    std::string in_dec_hidden_name_;
    std::string in_dec_symbol_name_;
    std::string out_dec_symbol_name_;
    std::string logits_name_;
    size_t end_token_;
    void check_net_names(const OutputsDataMap &output_info_decoder,
                         const InputsDataMap &input_info_decoder
                         ) const;
};

class DecoderNotFound {};
