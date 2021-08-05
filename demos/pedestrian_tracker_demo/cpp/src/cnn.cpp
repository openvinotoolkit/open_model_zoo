// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <inference_engine.hpp>

#include <utils/slog.hpp>
#include <utils/common.hpp>

CnnBase::CnnBase(const Config& config,
                 const InferenceEngine::Core & ie,
                 const std::string & deviceName) :
    config_(config), ie_(ie), deviceName_(deviceName) {}

void CnnBase::Load() {
    auto cnnNetwork = ie_.ReadNetwork(config_.path_to_model);

    const int currentBatchSize = cnnNetwork.getBatchSize();
    if (currentBatchSize != config_.max_batch_size)
        cnnNetwork.setBatchSize(config_.max_batch_size);

    InferenceEngine::InputsDataMap in;
    in = cnnNetwork.getInputsInfo();
    if (in.size() != 1) {
        throw std::runtime_error("Network should have only one input");
    }

    InferenceEngine::SizeVector inputDims = in.begin()->second->getTensorDesc().getDims();
    in.begin()->second->setPrecision(InferenceEngine::Precision::U8);
    input_blob_ = InferenceEngine::make_shared_blob<uint8_t>(InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, inputDims, InferenceEngine::Layout::NCHW));
    input_blob_->allocate();
    InferenceEngine::BlobMap inputs;
    inputs[in.begin()->first] = input_blob_;
    outInfo_ = cnnNetwork.getOutputsInfo();

    for (auto&& item : outInfo_) {
        InferenceEngine::SizeVector outputDims = item.second->getTensorDesc().getDims();
        auto outputLayout = item.second->getTensorDesc().getLayout();
        item.second->setPrecision(InferenceEngine::Precision::FP32);
        InferenceEngine::TBlob<float>::Ptr output =
            InferenceEngine::make_shared_blob<float>(InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outputDims, outputLayout));
        output->allocate();
        outputs_[item.first] = output;
    }

    executable_network_ = ie_.LoadNetwork(cnnNetwork, deviceName_);
    printExecNetworkInfo(executable_network_, config_.path_to_model, deviceName_, modelType);
    slog::info << "\tBatch size is set to " << config_.max_batch_size << slog::endl;
    infer_request_ = executable_network_.CreateInferRequest();
    infer_request_.SetInput(inputs);
    infer_request_.SetOutput(outputs_);
}

void CnnBase::InferBatch(
    const std::vector<cv::Mat>& frames,
    const std::function<void(const InferenceEngine::BlobMap&, size_t)>& fetch_results) const {
    const size_t batch_size = input_blob_->getTensorDesc().getDims()[0];

    size_t num_imgs = frames.size();
    for (size_t batch_i = 0; batch_i < num_imgs; batch_i += batch_size) {
        const size_t current_batch_size = std::min(batch_size, num_imgs - batch_i);
        for (size_t b = 0; b < current_batch_size; b++) {
            matU8ToBlob<uint8_t>(frames[batch_i + b], input_blob_, b);
        }

        infer_request_.Infer();

        fetch_results(outputs_, current_batch_size);
    }
}

void CnnBase::Infer(const cv::Mat& frame,
                    const std::function<void(const InferenceEngine::BlobMap&, size_t)>& fetch_results) const {
    InferBatch({frame}, fetch_results);
}

VectorCNN::VectorCNN(const Config& config,
                     const InferenceEngine::Core& ie,
                     const std::string & deviceName)
    : CnnBase(config, ie, deviceName) {
    Load();

    if (outputs_.size() != 1) {
        throw std::runtime_error("Demo supports topologies only with 1 output");
    }

    InferenceEngine::SizeVector dims = outInfo_.begin()->second->getTensorDesc().getDims();
    result_size_ = std::accumulate(std::next(dims.begin(), 1), dims.end(), 1, std::multiplies<int>());
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
                throw std::runtime_error("VectorCNN::Compute() Invalid blob '" + item.first + "'");
            }
            InferenceEngine::SizeVector ie_output_dims = blob->getTensorDesc().getDims();
            std::vector<int> blob_sizes(ie_output_dims.size(), 0);
            for (size_t i = 0; i < blob_sizes.size(); ++i) {
                blob_sizes[i] = ie_output_dims[i];
            }
            InferenceEngine::LockedMemory<const void> blobMapped =
                InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap();
            cv::Mat out_blob(blob_sizes, CV_32F, blobMapped.as<float*>());
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
