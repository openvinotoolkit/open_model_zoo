/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "cnn.hpp"

#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <inference_engine.hpp>

using namespace InferenceEngine;

CnnBase::CnnBase(const Config& config,
                 const InferenceEngine::InferencePlugin& plugin) :
    config_(config), net_plugin_(plugin) {}

void CnnBase::Load() {
    CNNNetReader net_reader;
    net_reader.ReadNetwork(config_.path_to_model);
    net_reader.ReadWeights(config_.path_to_weights);

    if (!net_reader.isParseSuccess()) {
        THROW_IE_EXCEPTION << "Cannot load model";
    }

    const int currentBatchSize = net_reader.getNetwork().getBatchSize();
    if (currentBatchSize != config_.max_batch_size)
        net_reader.getNetwork().setBatchSize(config_.max_batch_size);

    InferenceEngine::InputsDataMap in;
    in = net_reader.getNetwork().getInputsInfo();
    if (in.size() != 1) {
        THROW_IE_EXCEPTION << "Network should have only one input";
    }

    SizeVector inputDims = in.begin()->second->getTensorDesc().getDims();
    in.begin()->second->setInputPrecision(Precision::U8);
    input_blob_ = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, inputDims, Layout::NCHW));
    input_blob_->allocate();
    BlobMap inputs;
    inputs[in.begin()->first] = input_blob_;
    outInfo_ = net_reader.getNetwork().getOutputsInfo();

    for (auto&& item : outInfo_) {
        SizeVector outputDims = item.second->getTensorDesc().getDims();
        item.second->precision = Precision::FP32;
        TBlob<float>::Ptr output =
            make_shared_blob<float>(TensorDesc(Precision::FP32, outputDims, Layout::NC));
        output->allocate();
        outputs_[item.first] = output;
    }

    executable_network_ = net_plugin_.LoadNetwork(net_reader.getNetwork(), {});
    infer_request_ = executable_network_.CreateInferRequest();
    infer_request_.SetInput(inputs);
    infer_request_.SetOutput(outputs_);
}

void CnnBase::InferBatch(
    const std::vector<cv::Mat>& frames,
    std::function<void(const InferenceEngine::BlobMap&, size_t)> fetch_results) const {
    const size_t batch_size = input_blob_->getTensorDesc().getDims()[0];

    size_t num_imgs = frames.size();
    for (size_t batch_i = 0; batch_i < num_imgs; batch_i += batch_size) {
        const size_t current_batch_size = std::min(batch_size, num_imgs - batch_i);
        for (size_t b = 0; b < current_batch_size; b++) {
            matU8ToBlob<uint8_t>(frames[batch_i + b], input_blob_, b);
        }

        infer_request_.SetBatch(current_batch_size);
        infer_request_.Infer();

        fetch_results(outputs_, current_batch_size);
    }
}

void CnnBase::PrintPerformanceCounts() const {
    std::cout << "Performance counts for " << config_.path_to_model << std::endl << std::endl;
    ::printPerformanceCounts(infer_request_.GetPerformanceCounts(), std::cout, false);
}

void CnnBase::Infer(const cv::Mat& frame,
                    std::function<void(const InferenceEngine::BlobMap&, size_t)> fetch_results) const {
    InferBatch({frame}, fetch_results);
}

VectorCNN::VectorCNN(const Config& config,
                     const InferenceEngine::InferencePlugin& plugin)
    : CnnBase(config, plugin) {
    Load();

    if (outputs_.size() != 1) {
        THROW_IE_EXCEPTION << "Demo supports topologies only with 1 output";
    }
    InferenceEngine::SizeVector dims = outInfo_.begin()->second->dims;
    int num_dims = static_cast<int>(dims.size());

    result_size_ = 1;
    for (int i = 0; i < num_dims - 1; ++i) {
        result_size_ *= dims[i];
    }
}

void VectorCNN::Compute(const cv::Mat& frame,
                        cv::Mat* vector, cv::Size outp_shape) const {
    std::vector<cv::Mat> output;
    Compute({frame}, &output, outp_shape);
    *vector = output[0];
}

void VectorCNN::Compute(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* vectors,
                        cv::Size outp_shape) const {
    if (images.empty()) {
        return;
    }
    vectors->clear();
    auto results_fetcher = [vectors, outp_shape](const InferenceEngine::BlobMap& outputs, size_t batch_size) {
        for (auto&& item : outputs) {
            InferenceEngine::Blob::Ptr blob = item.second;
            if (blob == nullptr) {
                THROW_IE_EXCEPTION << "VectorCNN::Compute() Invalid blob '" << item.first << "'";
            }
            InferenceEngine::SizeVector ie_output_dims = blob->getTensorDesc().getDims();
            std::vector<int> blob_sizes(ie_output_dims.size(), 0);
            for (size_t i = 0; i < blob_sizes.size(); ++i) {
                blob_sizes[i] = ie_output_dims[i];
            }
            cv::Mat out_blob(blob_sizes, CV_32F, blob->buffer());
            for (size_t b = 0; b < batch_size; b++) {
                cv::Mat blob_wrapper(out_blob.size[1], 1, CV_32F,
                                     reinterpret_cast<void*>((out_blob.ptr<float>(0) + b * out_blob.size[1])));
                vectors->emplace_back();
                if (outp_shape != cv::Size())
                    blob_wrapper = blob_wrapper.reshape(1, {outp_shape.height, outp_shape.width});
                blob_wrapper.copyTo(vectors->back());
            }
        }
    };
    InferBatch(images, results_fetcher);
}
