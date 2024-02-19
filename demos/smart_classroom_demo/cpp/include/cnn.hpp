// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "openvino/openvino.hpp"

#include "utils/ocv_common.hpp"

/**
* @brief Base class of config for network
*/
struct CnnConfig {
    explicit CnnConfig(const std::string& path_to_model, const std::string& model_type = "") :
        m_path_to_model(path_to_model), m_model_type(model_type) {}
    /** @brief Path to model description */
    std::string m_path_to_model;
    /** @brief Model type*/
    std::string m_model_type;
    /** @brief Maximal size of batch */
    int m_max_batch_size{1};

    /** @brief OpenVINO Core instance */
    ov::Core m_core;
    /** @brief Device name */
    std::string m_deviceName;
};

/**
* @brief Base class of model
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

protected:
    /**
   * @brief Run model in batch mode
   *
   * @param frames Vector of input images
   * @param results_fetcher Callback to fetch inference results
   */
    void InferBatch(const std::vector<cv::Mat>& frames,
                    const std::function<void(const std::map<std::string, ov::Tensor>&, size_t)>& results_fetcher);

    /** @brief Config */
    Config m_config;
    /** @brief Model inputs info */
    ov::OutputVector m_inInfo;
    /** @brief Model outputs info */
    ov::OutputVector m_outInfo_;
    /** @brief Model layout */
    ov::Layout m_desired_layout;
    /** @brief Model input shape */
    ov::Shape m_modelShape;
    /** @brief Compled model */
    ov::CompiledModel m_compiled_model;
    /** @brief Inference request */
    ov::InferRequest m_infer_request;
    ov::Tensor m_in_tensor;
    /** @brief Names of output tensors */
    std::vector<std::string> m_output_tensors_names;
};

class VectorCNN : public CnnDLSDKBase {
public:
    explicit VectorCNN(const CnnConfig& config);

    void Compute(const cv::Mat& image,
                 cv::Mat* vector, cv::Size outp_shape = cv::Size());
    void Compute(const std::vector<cv::Mat>& images,
                 std::vector<cv::Mat>* vectors, cv::Size outp_shape = cv::Size());
    int maxBatchSize() const;
};

class AsyncAlgorithm {
public:
    virtual ~AsyncAlgorithm() {}
    virtual void enqueue(const cv::Mat& frame) = 0;
    virtual void submitRequest() = 0;
    virtual void wait() = 0;
};

template <typename T>
class AsyncDetection : public AsyncAlgorithm {
public:
    virtual std::vector<T> fetchResults() = 0;
};

template <typename T>
class NullDetection : public AsyncDetection<T> {
public:
    void enqueue(const cv::Mat&) override {}
    void submitRequest() override {}
    void wait() override {}
    std::vector<T> fetchResults() override { return {}; }
};

class BaseCnnDetection : public AsyncAlgorithm {
protected:
    std::shared_ptr<ov::InferRequest> m_request;
    const bool m_isAsync;
    std::string m_detectorName;

public:
    explicit BaseCnnDetection(bool isAsync = false) : m_isAsync(isAsync) {}

    void submitRequest() override {
        if (m_request == nullptr)
            return;
        if (m_isAsync) {
            m_request->start_async();
        } else {
            m_request->infer();
        }
    }

    void wait() override {
        if (!m_request || !m_isAsync)
            return;
        m_request->wait();
    }
};
