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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include <opencv2/opencv.hpp>
#include <samples/common.hpp>

#include <inference_engine.hpp>

/**
 * @brief Base class of config for network
 */
struct CnnConfig {
    explicit CnnConfig(const std::string& path_to_model,
                       const std::string& path_to_weights)
        : path_to_model(path_to_model), path_to_weights(path_to_weights) {}

    /** @brief Path to model description */
    std::string path_to_model;
    /** @brief Path to model weights */
    std::string path_to_weights;
    /** @brief Maximal size of batch */
    int max_batch_size{1};
};

/**
 * @brief Base class of network
 */
class CnnBase {
public:
    using Config = CnnConfig;

    /**
     * @brief Constructor
     */
    explicit CnnBase(const Config& config,
                     const InferenceEngine::InferencePlugin& plugin);

    /**
     * @brief Descructor
     */
    virtual ~CnnBase() {}

    /**
     * @brief Loads network
     */
    void Load();

    /**
     * @brief Prints performance report
     */
    void PrintPerformanceCounts() const;

protected:
    /**
     * @brief Run network
     *
     * @param frame Input image
     * @param results_fetcher Callback to fetch inference results
     */
    void Infer(const cv::Mat& frame,
               std::function<void(const InferenceEngine::BlobMap&, size_t)> results_fetcher) const;

    /**
     * @brief Run network in batch mode
     *
     * @param frames Vector of input images
     * @param results_fetcher Callback to fetch inference results
     */
    void InferBatch(const std::vector<cv::Mat>& frames,
                    std::function<void(const InferenceEngine::BlobMap&, size_t)> results_fetcher) const;

    /** @brief Config */
    Config config_;
    /** @brief IE plugin */
    InferenceEngine::InferencePlugin net_plugin_;
    /** @brief Net outputs info */
    InferenceEngine::OutputsDataMap outInfo_;
    /** @brief IE network */
    InferenceEngine::ExecutableNetwork executable_network_;
    /** @brief IE InferRequest */
    mutable InferenceEngine::InferRequest infer_request_;
    /** @brief Pointer to the pre-allocated input blob */
    mutable InferenceEngine::Blob::Ptr input_blob_;
    /** @brief Map of output blobs */
    InferenceEngine::BlobMap outputs_;
};

class VectorCNN : public CnnBase {
public:
    explicit VectorCNN(const CnnConfig& config,
                       const InferenceEngine::InferencePlugin& plugin);

    void Compute(const cv::Mat& image,
                 cv::Mat* vector, cv::Size outp_shape = cv::Size()) const;
    void Compute(const std::vector<cv::Mat>& images,
                 std::vector<cv::Mat>* vectors, cv::Size outp_shape = cv::Size()) const;

    int size() const { return result_size_; }

private:
    int result_size_;               ///< Length of result
};

