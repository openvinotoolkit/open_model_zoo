// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <chrono>
#include <map>
#include <string>

#include <samples/common.hpp>


void Cnn::Init(const std::string &model_path, Core & ie, const std::string & deviceName, const cv::Size &new_input_resolution) {
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- 1. Reading network ----------------------------------------------------
    CNNNetReader network_reader;
    network_reader.ReadNetwork(model_path);
    network_reader.ReadWeights(fileNameNoExt(model_path) + ".bin");
    network_reader.getNetwork().setBatchSize(1);

    model_path_ = model_path;

    CNNNetwork network = network_reader.getNetwork();

    // --------------------------- Changing input shape if it is needed ----------------------------------
    if (new_input_resolution != cv::Size()) {
        InputsDataMap inputInfo(network_reader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            THROW_IE_EXCEPTION << "The network should have only one input";
        }
        InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;

        SizeVector input_dims = inputInfoFirst->getInputData()->getTensorDesc().getDims();
        input_dims[2] = static_cast<size_t>(new_input_resolution.height);
        input_dims[3] = static_cast<size_t>(new_input_resolution.width);

        std::map<std::string, SizeVector> input_shapes;
        input_shapes[network.getInputsInfo().begin()->first] = input_dims;
        network.reshape(input_shapes);
    }

    // ---------------------------------------------------------------------------------------------------

    // --------------------------- Configuring input and output ------------------------------------------
    // ---------------------------   Preparing input blobs -----------------------------------------------
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;

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

    // --------------------------- Preparing input -------------------------------------------------------

    /* Resize manually and copy data from the image to the input blob */
    Blob::Ptr input = infer_request_.GetBlob(input_name);
    input_data_ = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

    is_initialized_ = true;
}

InferenceEngine::BlobMap Cnn::Infer(const cv::Mat &frame) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

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
                input_data_[ch * image_size + pid] = image.at<cv::Vec3f>(pid)[ch];
            }
        }
    } else if (channels_ == 1) {
        for (int pid = 0; pid < image_size; ++pid) {
            input_data_[pid] = image.at<float>(pid);
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
