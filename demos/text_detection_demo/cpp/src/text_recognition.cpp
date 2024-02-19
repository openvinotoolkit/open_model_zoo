// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <iostream>

#include "utils/image_utils.h"
#include "utils/ocv_common.hpp"
#include "text_recognition.hpp"

namespace  {

    constexpr size_t MAX_NUM_DECODER = 20;

    void ThrowNameNotFound(const std::string& name) {
        throw std::runtime_error("Name '" + name + "' does not exist in the model");
    };

    void softmax_and_choose(const std::vector<float>::const_iterator& begin, const std::vector<float>::const_iterator& end, int* argmax, float* prob) {
        auto max_element = std::max_element(begin, end);
        *argmax = static_cast<int>(std::distance(begin, max_element));
        float max_val = *max_element;
        double sum = 0;
        for (auto i = begin; i != end; i++) {
           sum += std::exp((*i) - max_val);
        }
        if (std::fabs(sum) < std::numeric_limits<double>::epsilon()) {
            throw std::logic_error("sum can't be equal to zero");
        }
        *prob = 1.0f / static_cast<float>(sum);
    }

    std::vector<float> softmax(const std::vector<float>::const_iterator& begin, const std::vector<float>::const_iterator& end) {
        std::vector<float> prob(end - begin, 0.f);
        std::transform(begin, end, prob.begin(), [](float x) { return std::exp(x); });
        float sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
        for (int i = 0; i < static_cast<int>(prob.size()); i++)
            prob[i] /= sum;
        return prob;
    }

    struct BeamElement {
        std::vector<int> sentence;   //!< The sequence of chars that will be a result of the beam element
        float prob_blank;            //!< The probability that the last char in CTC sequence
                                     //!< for the beam element is the special blank char
        float prob_not_blank;        //!< The probability that the last char in CTC sequence
                                     //!< for the beam element is NOT the special blank char

        float prob() const {         //!< The probability of the beam element.
            return prob_blank + prob_not_blank;
        }
    };
}  // namespace

std::string SimpleDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double* conf, int start_index) {
    std::string result = "";
    const int num_classes = alphabet.length();
    *conf = 1;

    for (std::vector<float>::const_iterator it = data.begin() + start_index * num_classes; it != data.end(); it += num_classes) {
        int argmax;
        float prob;

        softmax_and_choose(it, it + num_classes, &argmax, &prob);
        (*conf) *= prob;
        auto symbol = alphabet[argmax];
        if (symbol != pad_symbol)
            result += symbol;
        else
            break;
    }

    return result;
}

std::string CTCGreedyDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double* conf) {
    std::string result = "";
    bool prev_pad = false;
    *conf = 1;

    const int num_classes = alphabet.length();
    for (std::vector<float>::const_iterator it = data.begin(); it != data.end(); it += num_classes) {
      int argmax;
      float prob;

      softmax_and_choose(it, it + num_classes, &argmax, &prob);

      (*conf) *= prob;

      auto symbol = alphabet[argmax];
      if (symbol != pad_symbol) {
          if (result.empty() || prev_pad || (!result.empty() && symbol != result.back())) {
            prev_pad = false;
            result += symbol;
          }
      } else {
        prev_pad = true;
      }
    }

    return result;
}

std::string CTCBeamSearchDecoder(const std::vector<float>& data, const std::string& alphabet, char pad_symbol, double* conf, int bandwidth) {
    const int num_classes = alphabet.length();

    std::vector<BeamElement> curr;
    std::vector<BeamElement> last;

    last.push_back(BeamElement{std::vector<int>(), 1.f, 0.f});

    for (std::vector<float>::const_iterator it = data.begin(); it != data.end(); it += num_classes) {
        curr.clear();

        std::vector<float> prob = softmax(it, it + num_classes);

        for(const auto& candidate: last) {
            float prob_not_blank = 0.f;
            const std::vector<int>& candidate_sentence = candidate.sentence;
            if (!candidate_sentence.empty()) {
                int n = candidate_sentence.back();
                prob_not_blank = candidate.prob_not_blank * prob[n];
            }
            float prob_blank = candidate.prob() * prob[num_classes - 1];

            auto cmp = [&candidate_sentence](const BeamElement& n) {
                return n.sentence == candidate_sentence;
            };
            auto check_res = std::find_if(curr.begin(), curr.end(), cmp);
            if (check_res == std::end(curr)) {
                curr.push_back(BeamElement{candidate.sentence, prob_blank, prob_not_blank});
            } else {
                check_res->prob_not_blank  += prob_not_blank;
                if (check_res->prob_blank != 0.f) {
                    throw std::logic_error("Probability that the last char in CTC-sequence is the special blank char must be zero here");
                }
                check_res->prob_blank = prob_blank;
            }

            for (int i = 0; i < num_classes - 1; i++) {
                auto extend = candidate_sentence;
                extend.push_back(i);

                if (candidate_sentence.size() > 0 && candidate.sentence.back() == i) {
                    prob_not_blank = prob[i] * candidate.prob_blank;
                } else {
                    prob_not_blank = prob[i] * candidate.prob();
                }

                auto check_res = std::find_if(curr.begin(), curr.end(), [&extend](const BeamElement &n) {
                    return n.sentence == extend;
                });

                if (check_res == std::end(curr)) {
                    curr.push_back(BeamElement{extend, 0.f, prob_not_blank});
                } else {
                    check_res->prob_not_blank += prob_not_blank;
                }
            }
        }

        sort(curr.begin(), curr.end(), [](const BeamElement &a, const BeamElement &b) -> bool {
            return a.prob() > b.prob();
        });

        last.clear();
        int num_to_copy = std::min(bandwidth, static_cast<int>(curr.size()));
        for (int b = 0; b < num_to_copy; b++) {
            last.push_back(curr[b]);
        }
    }

    *conf = last[0].prob();
    std::string result = "";
    for (const auto& idx: last[0].sentence) {
        result += alphabet[idx];
    }

    return result;
}

TextRecognizer::TextRecognizer(
    const std::string& model_path, const std::string& model_type, const std::string& deviceName,
    ov::Core& core,
    const std::string& out_enc_hidden_name, const std::string& out_dec_hidden_name,
    const std::string& in_dec_hidden_name, const std::string& features_name,
    const std::string& in_dec_symbol_name, const std::string& out_dec_symbol_name,
    const std::string& logits_name, size_t end_token, bool use_auto_resize) :
    Cnn(model_path, model_type, deviceName, core, {}, use_auto_resize),
    m_isCompositeModel(false),
    m_features_name(features_name), m_out_enc_hidden_name(out_enc_hidden_name),
    m_out_dec_hidden_name(out_dec_hidden_name), m_in_dec_hidden_name(in_dec_hidden_name),
    m_in_dec_symbol_name(in_dec_symbol_name), m_out_dec_symbol_name(out_dec_symbol_name),
    m_logits_name(logits_name), m_end_token(end_token)
{
    // text-recognition-0015/0016 consist from encoder and decoder models, both have to be read
    if (model_path.find("encoder") == std::string::npos)
        return;

    std::string model_decoder_path(model_path);
    while (model_decoder_path.find("encoder") != std::string::npos)
        model_decoder_path = model_decoder_path.replace(model_decoder_path.find("encoder"), 7, "decoder");

    m_isCompositeModel = true;

    slog::info << "Reading model: " << model_decoder_path << slog::endl;
    std::shared_ptr<ov::Model> model_decoder = core.read_model(model_decoder_path);
    logBasicModelInfo(model_decoder);

    check_model_names(model_decoder->inputs(), model_decoder->outputs());

    // Loading model to the device
    ov::CompiledModel compiled_model_decoder = core.compile_model(model_decoder, deviceName);
    logCompiledModelInfo(compiled_model_decoder, model_decoder_path, deviceName, model_type);

    // Creating infer request
    m_infer_request_decoder = compiled_model_decoder.create_infer_request();
}

std::map<std::string, ov::Tensor> TextRecognizer::Infer(const cv::Mat& frame) {
    std::chrono::steady_clock::time_point begin_enc = std::chrono::steady_clock::now();

    cv::Mat image;
    if (m_channels == 1) {
        cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);
    } else {
        image = frame;
    }

    ov::Tensor tensor = m_infer_request.get_tensor(m_input_name);
    if (!use_auto_resize) {
        image = resizeImageExt(image, m_input_size.width, m_input_size.height);
    }
    m_infer_request.set_tensor(m_input_name, wrapMat2Tensor(image));

    m_infer_request.infer();

    // Processing output
    std::map<std::string, ov::Tensor> output_tensors;
    for (const auto& output_name : m_output_names) {
        output_tensors[output_name] = m_infer_request.get_tensor(output_name);
    }

    std::chrono::steady_clock::time_point end_enc = std::chrono::steady_clock::now();
    m_time_elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(end_enc - begin_enc).count();
    m_ncalls++;

    if(!m_isCompositeModel)
        return output_tensors;

    // measure decoder part (in case of composite model)
    std::chrono::steady_clock::time_point begin_dec = std::chrono::steady_clock::now();

    // Processing encoder output
    // tensors here are set for concrete network
    // in case of different network this needs to be changed or generalized
    ov::Tensor features_tensor = output_tensors[m_features_name];
    m_infer_request_decoder.set_tensor(m_features_name, features_tensor);

    ov::Tensor out_enc_hidden_tensor = output_tensors[m_out_enc_hidden_name];
    m_infer_request_decoder.set_tensor(m_in_dec_hidden_name, out_enc_hidden_tensor);

    float* input_data_decoder = m_infer_request_decoder.get_tensor(m_in_dec_symbol_name).data<float>();

    input_data_decoder[0] = 0;
    size_t num_classes = m_infer_request_decoder.get_tensor(m_out_dec_symbol_name).get_size();

    ov::Tensor targets(ov::element::f32, { 1, MAX_NUM_DECODER, num_classes });
    float* data_targets = targets.data<float>();

    for (size_t num_decoder = 0; num_decoder < MAX_NUM_DECODER; num_decoder++) {
        m_infer_request_decoder.infer();

        const float* output_data_decoder = m_infer_request_decoder.get_tensor(m_out_dec_symbol_name).data<float>();

        auto max_elem_vector = std::max_element(output_data_decoder, output_data_decoder + num_classes);
        auto argmax = static_cast<size_t>(std::distance(output_data_decoder, max_elem_vector));
        for (size_t i = 0; i < num_classes; i++)
            data_targets[num_decoder * num_classes + i] = output_data_decoder[i];

        if (m_end_token == argmax)
            break;

        input_data_decoder[0] = float(argmax);

        ov::Tensor out_enc_hidden_tensor = m_infer_request_decoder.get_tensor(m_out_enc_hidden_name);
        m_infer_request_decoder.set_tensor(m_in_dec_hidden_name, out_enc_hidden_tensor);
    }

    std::chrono::steady_clock::time_point end_dec = std::chrono::steady_clock::now();
    m_time_elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(end_dec - begin_dec).count();

    return { {m_logits_name, targets} };
}

void TextRecognizer::check_model_names(
    const ov::OutputVector& input_info_decoder,
    const ov::OutputVector& output_info_decoder) const
{
    if (std::find(m_output_names.begin(), m_output_names.end(), m_out_enc_hidden_name) == m_output_names.end())
        ThrowNameNotFound(m_out_enc_hidden_name);

    if (std::find(m_output_names.begin(), m_output_names.end(), m_features_name) == m_output_names.end())
        ThrowNameNotFound(m_features_name);

    auto in_dec_hidden_name = [&](ov::Output<ov::Node> output) { return output.get_any_name() == m_in_dec_hidden_name; };
    if (input_info_decoder.end() == std::find_if(input_info_decoder.begin(), input_info_decoder.end(), in_dec_hidden_name))
        ThrowNameNotFound(m_in_dec_hidden_name);

    auto features_name = [&](ov::Output<ov::Node> output) { return output.get_any_name() == m_features_name; };
    if (input_info_decoder.end() == std::find_if(input_info_decoder.begin(), input_info_decoder.end(), features_name))
        ThrowNameNotFound(m_features_name);

    auto in_dec_symbol_name = [&](ov::Output<ov::Node> output) { return output.get_any_name() == m_in_dec_symbol_name; };
    if (input_info_decoder.end() == std::find_if(input_info_decoder.begin(), input_info_decoder.end(), in_dec_symbol_name))
        ThrowNameNotFound(m_in_dec_symbol_name);

    auto out_dec_hidden_name = [&](ov::Output<ov::Node> output) { return output.get_any_name() == m_out_dec_hidden_name; };
    if (output_info_decoder.end() == std::find_if(output_info_decoder.begin(), output_info_decoder.end(), out_dec_hidden_name))
        ThrowNameNotFound(m_out_dec_hidden_name);

    auto out_dec_symbol_name = [&](ov::Output<ov::Node> output) { return output.get_any_name() == m_out_dec_symbol_name; };
    if (output_info_decoder.end() == std::find_if(output_info_decoder.begin(), output_info_decoder.end(), out_dec_symbol_name))
        ThrowNameNotFound(m_out_dec_symbol_name);
}
