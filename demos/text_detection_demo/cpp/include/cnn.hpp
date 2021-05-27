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
    Cnn():is_initialized_(false), channels_(0), time_elapsed_(0), ncalls_(0) {}

    virtual void Init(const std::string &model_path, Core & ie, const std::string & deviceName,
              const cv::Size &new_input_resolution = cv::Size());

    virtual InferenceEngine::BlobMap Infer(const cv::Mat &frame);

    bool is_initialized() const {return is_initialized_;}

    size_t ncalls() const {return ncalls_;}
    double time_elapsed() const {return time_elapsed_;}
    virtual void setInOutNames(const std::string out_enc_hidden_name,
                               const std::string out_dec_hidden_name,
                               const std::string in_dec_hidden_name,
                               const std::string features_name,
                               const std::string in_dec_symbol_name,
                               const std::string out_dec_symbol_name,
                               const std::string logits_name) {};
    const cv::Size& input_size() const {return input_size_;}

  protected:
    bool is_initialized_;
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

    void Init(const std::string &model_path, Core & ie, const std::string & deviceName,
              const cv::Size &new_input_resolution = cv::Size()) override;

    InferenceEngine::BlobMap Infer(const cv::Mat &frame) override;
    void setInOutNames(const std::string out_enc_hidden_name,
                      const std::string out_dec_hidden_name,
                      const std::string in_dec_hidden_name,
                      const std::string features_name,
                      const std::string in_dec_symbol_name,
                      const std::string out_dec_symbol_name,
                      const std::string logits_name) override;
  private:
    InferRequest infer_request_encoder_;
    InferRequest infer_request_decoder_;
    std::vector<std::string> output_names_encoder;
    std::vector<std::string> output_names_decoder;
    std::string features_name_;
    std::string out_enc_hidden_name_;
    std::string out_dec_hidden_name_;
    std::string in_dec_hidden_name_;
    std::string in_dec_symbol_name_;
    std::string out_dec_symbol_name_;
    std::string logits_name_;
    void check_net_names(std::vector<std::string> output_names_encoder,
                                               std::vector<std::string> input_names_decoder,
                                               std::vector<std::string> output_names_decoder);
};

class NameNotExist : public std::exception {
  private:
	std::string error_message;
  public:
	explicit NameNotExist(const std::string& name) {
		error_message = std::string("Name '") + name + std::string("' does not exist in the network");
	};
	const char * what () const noexcept override {
		return error_message.c_str();
	};
};

class DecoderNotFound : public std::exception {
  public:
	explicit DecoderNotFound() { };
	const char * what () const noexcept override {
		return "Decoder not found";
	};
};
