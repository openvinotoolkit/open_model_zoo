// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>


#include <openvino/openvino.hpp>
#include <utils/ocv_common.hpp>

/**
 * @brief Base class of config for network
 */
struct CnnConfigTracker {
    explicit CnnConfigTracker(const std::string& path_to_model)
        : path_to_model(path_to_model) {}

    /** @brief Path to model description */
    std::string path_to_model;
    /** @brief Maximal size of batch */
    int max_batch_size{1};
};

/**
 * @brief Base class of network
 */
class CnnBase {
public:
    using Config = CnnConfigTracker;

    /**
     * @brief Constructor
     */
    CnnBase(const Config& config,
            const ov::Core& core,
            const std::string & deviceName);

    /**
     * @brief Descructor
     */
    virtual ~CnnBase() {}

    /**
     * @brief Loads network
     */
    void Load();

    const std::string modelType = "Person Re-Identification";

protected:
    /**
     * @brief Run network in batch mode
     *
     * @param frames Vector of input images
     * @param results_fetcher Callback to fetch inference results
     */
    void InferBatch(const std::vector<cv::Mat>& frames,
                    const std::function<void(const ov::Tensor&, size_t)>& results_fetcher) const;

    /** @brief Config */
    Config config;
    /** @brief OpenVINO instance */
    ov::Core core;
    /** @brief device */
    std::string device_name;
    /** @brief Model inputs info */
    ov::OutputVector inputs;
    /** @brief Model outputs info */
    ov::OutputVector outputs;
     /** @brief Model input layout */
    ov::Layout input_layout;
    /** @brief Compiled model */
    ov::CompiledModel compiled_model;
    /** @brief Inference Request */
    mutable ov::InferRequest infer_request;
    /** @brief Input Tensor */
    mutable ov::Tensor input_tensor;
    /** @brief Input Tensor shape */
    ov::Shape input_shape;
    /** @brief Output tensor */
    ov::Tensor output_tensor;
    /** @brief Input Tensor shape */
    ov::Shape output_shape;
};

class VectorCNN : public CnnBase {
public:
    VectorCNN(const CnnConfigTracker& config,
              const ov::Core & core,
              const std::string & deviceName);

    void Compute(const cv::Mat& image,
                 cv::Mat* vector, cv::Size outp_shape = cv::Size()) const;
    void Compute(const std::vector<cv::Mat>& images,
                 std::vector<cv::Mat>* vectors, cv::Size outp_shape = cv::Size()) const;

    int size() const { return result_size; }

private:
    int result_size;  /// Length of result
};
