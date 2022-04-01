// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

/**
 * @brief Base class of config for model
 */
struct ModelConfigTracker {
    explicit ModelConfigTracker(const std::string& path_to_model) : path_to_model(path_to_model) {}

    /** @brief Path to model description */
    std::string path_to_model;
    /** @brief Maximal size of batch */
    int max_batch_size{1};
};

/**
 * @brief Base class of model
 */
class BaseModel {
public:
    using Config = ModelConfigTracker;

    /**
     * @brief Constructor
     */
    BaseModel(const Config& config, const ov::Core& core, const std::string& deviceName);

    /**
     * @brief Descructor
     */
    virtual ~BaseModel() {}

    /**
     * @brief Loads model
     */
    void Load();

    const std::string modelType = "Person Re-Identification";

protected:
    /**
     * @brief Run model in batch mode
     *
     * @param frames Vector of input images
     * @param results_fetcher Callback to fetch inference results
     */
    void InferBatch(const std::vector<cv::Mat>& frames,
                    const std::function<void(const ov::Tensor&, size_t)>& results_fetcher) const;

    /** @brief Config */
    Config config;
    /** @brief OpenVINO Core instance */
    ov::Core core;
    /** @brief device */
    std::string device_name;
    /** @brief Model input layout */
    ov::Layout input_layout;
    /** @brief Compiled model */
    ov::CompiledModel compiled_model;
    /** @brief Inference Request */
    mutable ov::InferRequest infer_request;
    /** @brief Input tensor */
    mutable ov::Tensor input_tensor;
    /** @brief Input tensor shape */
    ov::Shape input_shape;
    /** @brief Output tensor */
    ov::Tensor output_tensor;
    /** @brief Input tensor shape */
    ov::Shape output_shape;
};

class VectorCNN : public BaseModel {
public:
    VectorCNN(const ModelConfigTracker& config, const ov::Core& core, const std::string& deviceName);

    void Compute(const cv::Mat& image, cv::Mat* vector, cv::Size outp_shape = cv::Size()) const;
    void Compute(const std::vector<cv::Mat>& images,
                 std::vector<cv::Mat>* vectors,
                 cv::Size outp_shape = cv::Size()) const;

    int size() const {
        return result_size;
    }

private:
    int result_size;  // Length of result
};
