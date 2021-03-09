// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include <samples/ocv_common.hpp>

#include <inference_engine.hpp>

/**
* @brief Base class of config for network
*/
struct CnnConfig {
    explicit CnnConfig(const std::string& path_to_model)
        : path_to_model(path_to_model) {}

    /** @brief Path to model description */
    std::string path_to_model;
    /** @brief Maximal size of batch */
    int max_batch_size{1};

    /** @brief Inference Engine */
    InferenceEngine::Core ie;
    /** @brief Device name */
    std::string deviceName;
};

/**
* @brief Base class of network
*/
class CnnDLSDKBase {
public:
    using Config = CnnConfig;

    /**
   * @brief Constructor
   */
    explicit CnnDLSDKBase(const Config& config);

    /**
   * @brief Descructor
   */
    ~CnnDLSDKBase() {}

    /**
   * @brief Loads network
   */
    void Load();

    /**
    * @brief Prints performance report
    */
    void PrintPerformanceCounts(std::string fullDeviceName) const;

protected:
    /**
   * @brief Run network
   *
   * @param frame Input image
   * @param results_fetcher Callback to fetch inference results
   */
    void Infer(const cv::Mat& frame,
               const std::function<void(const InferenceEngine::BlobMap&, size_t)>& results_fetcher) const;

    /**
   * @brief Run network in batch mode
   *
   * @param frames Vector of input images
   * @param results_fetcher Callback to fetch inference results
   */
    void InferBatch(const std::vector<cv::Mat>& frames,
                    const std::function<void(const InferenceEngine::BlobMap&, size_t)>& results_fetcher) const;

    /** @brief Config */
    Config config_;
    /** @brief Net inputs info */
    InferenceEngine::InputsDataMap inInfo_;
    /** @brief Net outputs info */
    InferenceEngine::OutputsDataMap outInfo_;
    /** @brief IE network */
    InferenceEngine::ExecutableNetwork executable_network_;
    /** @brief IE InferRequest */
    mutable InferenceEngine::InferRequest infer_request_;
    /** @brief Name of the input blob input blob */
    std::string input_blob_name_;
    /** @brief Names of output blobs */
    std::vector<std::string> output_blobs_names_;
};

class VectorCNN : public CnnDLSDKBase {
public:
    explicit VectorCNN(const CnnConfig& config);

    void Compute(const cv::Mat& image,
                 cv::Mat* vector, cv::Size outp_shape = cv::Size()) const;
    void Compute(const std::vector<cv::Mat>& images,
                 std::vector<cv::Mat>* vectors, cv::Size outp_shape = cv::Size()) const;
};

class AsyncAlgorithm {
public:
    virtual ~AsyncAlgorithm() {}
    virtual void enqueue(const cv::Mat &frame) = 0;
    virtual void submitRequest() = 0;
    virtual void wait() = 0;
    virtual void printPerformanceCounts(const std::string &fullDeviceName) = 0;
};

template <typename T>
class AsyncDetection : public AsyncAlgorithm {
public:
    virtual std::vector<T> fetchResults() = 0;
};

template <typename T>
class NullDetection : public AsyncDetection<T> {
public:
    void enqueue(const cv::Mat &) override {}
    void submitRequest() override {}
    void wait() override {}
    void printPerformanceCounts(const std::string &) override {}
    std::vector<T> fetchResults() override { return {}; }
};

class BaseCnnDetection : public AsyncAlgorithm {
protected:
    InferenceEngine::InferRequest::Ptr request;
    const bool isAsync;
    std::string topoName;

public:
    explicit BaseCnnDetection(bool isAsync = false) :
                              isAsync(isAsync) {}

    void submitRequest() override {
        if (request == nullptr) return;
        if (isAsync) {
            request->StartAsync();
        } else {
            request->Infer();
        }
    }

    void wait() override {
        if (!request || !isAsync) return;
        request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    }

    void printPerformanceCounts(const std::string &fullDeviceName) override {
        std::cout << "Performance counts for " << topoName << std::endl << std::endl;
        ::printPerformanceCounts(*request, std::cout, fullDeviceName, false);
    }
};
