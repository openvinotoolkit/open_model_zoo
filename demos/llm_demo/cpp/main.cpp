// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common.h>  // llama.cpp
#include <llama.h>
#include <openvino/openvino.hpp>
#include <utils/slog.hpp>

namespace {
void print_token(const llama_model& vocab, llama_token out_token) {
    std::array<char, 13> decoded;
    int length = llama_token_to_piece(&vocab, out_token, decoded.data(), decoded.size());
    if (length <= 0) {
        throw std::runtime_error("Unexpected number of chars for the token: " + std::to_string(length));
    }
    for (int idx = 0; idx < length; ++idx) {
        std::cout << decoded[idx];
    }
    std::cout << std::flush;
}
}

int main(int argc, char* argv[]) try {
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage : "} + argv[0] + " <model_path> <vocab_path> '<prompt>'");
    }
    llama_model_params params;
    params.vocab_only = true;
    struct LlamaDeleter {void operator()(llama_model* ptr) noexcept {llama_free_model(ptr);}};
    std::unique_ptr<llama_model, LlamaDeleter> vocab{llama_load_model_from_file(argv[2], params)};
    if (!vocab) {
        throw std::runtime_error("Failed to load vocab");
    }
    constexpr bool add_bos = true;
    std::vector<llama_token> prompt = llama_tokenize(vocab.get(), argv[3], add_bos);
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<std::string, ov::PartialShape> shapes = {
        {"input_ids", ov::PartialShape{
            BATCH_SIZE, {1, std::numeric_limits<ov::Dimension::value_type>::max()}
        }},
        {"attention_mask", ov::PartialShape{
            BATCH_SIZE, {1, std::numeric_limits<ov::Dimension::value_type>::max()}
        }}
    };
    for (const ov::Output<const ov::Node>& input : model->inputs()) {
        for (const std::string& name : input.get_names()) {
            if (name.find("past_key_values") == 0) {
                ov::PartialShape shape = input.get_partial_shape();
                shape[0] = BATCH_SIZE;
                shapes.emplace(name, shape);
                break;
            }
        }
    }
    model->reshape(shapes);
    ov::InferRequest ireq = core.compile_model(model, "CPU", {ov::cache_dir("llm-cache")}).create_infer_request();
    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, prompt.size()});
    std::copy(prompt.begin(), prompt.end(), ireq.get_tensor("input_ids").data<int64_t>());
    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, prompt.size()});
    std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), prompt.size(), 1);
    for (const ov::Output<const ov::Node>& input : model->inputs()) {
        for (const std::string& name : input.get_names()) {
            if (name.find("past_key_values") == 0) {
                ireq.get_tensor(input).set_shape(input.get_partial_shape().get_min_shape());
                break;
            }
        }
    }
    ireq.infer();
    size_t n_vocab = llama_n_vocab(vocab.get());
    if (ireq.get_tensor("logits").get_shape().back() != n_vocab) {
        throw std::runtime_error("Model and vocab number of tokens don't match");
    }
    float* logits = ireq.get_tensor("logits").data<float>() + (prompt.size() - 1) * n_vocab;
    size_t out_token = std::max_element(logits, logits + n_vocab) - logits;

    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("attention_mask").data<int64_t>()[0] = 1;
    constexpr size_t SPECIAL_EOS_ID = 2;
    while (out_token != SPECIAL_EOS_ID) {
        for (const ov::Output<const ov::Node>& input : model->inputs()) {
            for (const std::string& name : input.get_names()) {
                if (name.find("past_key_values") == 0) {
                    ireq.set_tensor(input, ireq.get_tensor("present" + name.substr(15)));
                    break;
                }
            }
        }
        ireq.get_tensor("input_ids").data<int64_t>()[0] = out_token;
        ireq.start_async();
        print_token(*vocab, out_token);
        ireq.wait();
        logits = ireq.get_tensor("logits").data<float>();
        out_token = std::max_element(logits, logits + n_vocab) - logits;
    }
    std::cout << '\n';
} catch (const std::exception& error) {
    slog::err << error.what() << slog::endl;
    return 1;
} catch (...) {
    slog::err << "Non-exception object thrown" << slog::endl;
    return 1;
}
