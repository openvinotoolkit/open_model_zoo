// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <chrono>
#include <map>
#include <string>

#include <utils/common.hpp>

#define MAX_NUM_DECODER 20

void Cnn::Init(const std::string &model_path, Core & ie, const std::string & deviceName, const cv::Size &new_input_resolution) {
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- 1. Reading network ----------------------------------------------------
    auto network = ie.ReadNetwork(model_path);

    // --------------------------- Changing input shape if it is needed ----------------------------------
    InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::runtime_error("The network should have only one input");
    }
    InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;

    SizeVector input_dims = inputInfoFirst->getInputData()->getTensorDesc().getDims();
    input_dims[0] = 1;
    if (new_input_resolution != cv::Size()) {
        input_dims[2] = static_cast<size_t>(new_input_resolution.height);
        input_dims[3] = static_cast<size_t>(new_input_resolution.width);
    }

    std::map<std::string, SizeVector> input_shapes;
    input_shapes[network.getInputsInfo().begin()->first] = input_dims;
    network.reshape(input_shapes);

    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Configuring input and output ------------------------------------------
    // ---------------------------   Preparing input blobs -----------------------------------------------
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    input_name_ = network.getInputsInfo().begin()->first;

    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::FP32);

    channels_ = input_info->getTensorDesc().getDims()[1];
    input_size_ = cv::Size(input_info->getTensorDesc().getDims()[3], input_info->getTensorDesc().getDims()[2]);

    // ---------------------------   Preparing output blobs ----------------------------------------------

    OutputsDataMap output_info(network.getOutputsInfo());
    for (auto output : output_info) {
        output_names_.emplace_back(output.first);
    }

    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Loading model to the device -------------------------------------------
    ExecutableNetwork executable_network = ie.LoadNetwork(network, deviceName);
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Creating infer request ------------------------------------------------
    infer_request_ = executable_network.CreateInferRequest();
    // ---------------------------------------------------------------------------------------------------

    is_initialized_ = true;
}

InferenceEngine::BlobMap Cnn::Infer(const cv::Mat &frame) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    /* Resize manually and copy data from the image to the input blob */
    InferenceEngine::LockedMemory<void> inputMapped =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(infer_request_.GetBlob(input_name_))->wmap();
    float* input_data = inputMapped.as<float *>();

    cv::Mat image;
    if (channels_ == 1) {
         cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);
    } else {
        image = frame.clone();
    }

    image.convertTo(image, CV_32F);
    cv::resize(image, image, input_size_);

    int image_size = input_size_.area();

    if (channels_ == 3) {
        for (int pid = 0; pid < image_size; ++pid) {
            for (int ch = 0; ch < channels_; ++ch) {
                input_data[ch * image_size + pid] = image.at<cv::Vec3f>(pid)[ch];
            }
        }
    } else if (channels_ == 1) {
        for (int pid = 0; pid < image_size; ++pid) {
            input_data[pid] = image.at<float>(pid);
        }
    }
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Doing inference -------------------------------------------------------
    /* Running the request synchronously */
    infer_request_.Infer();
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Processing output -----------------------------------------------------

    InferenceEngine::BlobMap blobs;
    for (const auto &output_name : output_names_) {
        blobs[output_name] = infer_request_.GetBlob(output_name);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    time_elapsed_ += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    ncalls_++;

    return blobs;
}


void EncoderDecoderCNN::Init(const std::string &model_path, Core & ie, const std::string & deviceName, const cv::Size &new_input_resolution) {
    // ---------------------------------------------------------------------------------------------------
    // --------------------------- 0. checking paths -----------------------------------------------------
    std::string model_path_decoder = model_path;
    model_path_decoder = model_path_decoder.replace(model_path_decoder.find("encoder"), 7, "decoder");
    auto network_encoder = ie.ReadNetwork(model_path);
    auto network_decoder = ie.ReadNetwork(model_path_decoder);
    InputsDataMap inputInfo(network_encoder.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::runtime_error("The network_encoder should have only one input");
    }
    inputInfo = network_decoder.getInputsInfo();
    for (auto input : inputInfo) {
        input_names_decoder.emplace_back(input.first);
    }
    OutputsDataMap outputInfo(network_encoder.getOutputsInfo());
    for (auto output : outputInfo) {
        output_names_encoder.emplace_back(output.first);
    }
    outputInfo = network_decoder.getOutputsInfo();
    for (auto output : outputInfo) {
        output_names_decoder.emplace_back(output.first);
    }

    // ---------------------------------------------------------------------------------------------------
    InputInfo::Ptr input_info = network_encoder.getInputsInfo().begin()->second;
    input_info_decoder_ = network_decoder.getInputsInfo();
    input_name_ = network_encoder.getInputsInfo().begin()->first;

    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::FP32);

    channels_ = input_info->getTensorDesc().getDims()[1];
    input_size_ = cv::Size(input_info->getTensorDesc().getDims()[3], input_info->getTensorDesc().getDims()[2]);
    // --------------------------- Loading model to the device -------------------------------------------
    ExecutableNetwork executable_network_encoder = ie.LoadNetwork(network_encoder, deviceName);
    ExecutableNetwork executable_network_decoder = ie.LoadNetwork(network_decoder, deviceName);
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Creating infer request ------------------------------------------------
    infer_request_encoder_ = executable_network_encoder.CreateInferRequest();
    infer_request_decoder_ = executable_network_decoder.CreateInferRequest();
    // ---------------------------------------------------------------------------------------------------

    is_initialized_ = true;
}

InferenceEngine::BlobMap EncoderDecoderCNN::Infer(const cv::Mat &frame) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    /* Resize manually and copy data from the image to the input blob */
    InferenceEngine::LockedMemory<void> inputMapped =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(infer_request_encoder_.GetBlob(input_name_))->wmap();
    float* input_data = inputMapped.as<float *>();

    cv::Mat image;
    if (channels_ == 1) {
         cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);
    } else {
        image = frame.clone();
    }

    image.convertTo(image, CV_32F);
    cv::resize(image, image, input_size_);

    int image_size = input_size_.area();

    if (channels_ == 3) {
        for (int pid = 0; pid < image_size; ++pid) {
            for (int ch = 0; ch < channels_; ++ch) {
                input_data[ch * image_size + pid] = image.at<cv::Vec3f>(pid)[ch];
            }
        }
    } else if (channels_ == 1) {
        for (int pid = 0; pid < image_size; ++pid) {
            input_data[pid] = image.at<float>(pid);
        }
    }
    // --------------------------- Doing inference -------------------------------------------------------
    /* Running the request synchronously */
    infer_request_encoder_.Infer();
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Processing output encoder -----------------------------------------------------

    InferenceEngine::BlobMap encoder_blobs;
    for (const auto &output_name : output_names_encoder) {
        encoder_blobs[output_name] = infer_request_encoder_.GetBlob(output_name);
    }
    // blobs here are set for concrete network
    // in case of different network this needs to be changed or generalized
    infer_request_decoder_.SetBlob("features", encoder_blobs["features"]);
    infer_request_decoder_.SetBlob("hidden", encoder_blobs["hidden"]);
    InferenceEngine::TBlob<float>::Ptr tgt;
    tgt = InferenceEngine::make_shared_blob<float>(
        InferenceEngine::TensorDesc(Precision::U8, input_info_decoder_["decoder_input"]->getTensorDesc().getDims(),
        input_info_decoder_["decoder_input"]->getTensorDesc().getLayout()));
    tgt->allocate();
    tgt[0][0] = 0.0;
    // infer_request_decoder_.SetBlob("decoder_input", tgt);
    // InferenceEngine::BlobMap decoder_blobs;
    // auto targets = std::vector<InferenceEngine::MemoryBlob::Ptr> {};
    // for (size_t num_decoder = 0; num_decoder < MAX_NUM_DECODER; num_decoder ++) {
    //     infer_request_decoder_.Infer();
    //     for (const auto &output_name : output_names_decoder) {
    //         decoder_blobs[output_name] = infer_request_decoder_.GetBlob(output_name);
    //     }
    //     // argmax:
    //     auto result = std::max_element(decoder_blobs["decoder_output"].begin(), decoder_blobs["decoder_output"].begin());
    //     auto token = std::distance(decoder_blobs["decoder_output"].begin(), result);
    //     tgt.CopyVectorToBlob(tgt, std::vector<std::vector<float>>{{token}});
    //     targets.push_back(tgt);
    //     infer_request_encoder_.SetBlob("features", encoder_blobs["features"]);
    //     infer_request_encoder_.SetBlob("hidden", decoder_blobs["hidden"]);
    //     infer_request_encoder_.SetBlob("decoder_input", tgt);
    // }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    time_elapsed_ += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    ncalls_++;
    // InferenceEngine::BlobMap blobs;
    return encoder_blobs;
    // blobs['targets'] =
    // return blobs;
}

