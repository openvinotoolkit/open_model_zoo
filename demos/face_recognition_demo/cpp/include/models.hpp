// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "async_queue.hpp"

#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <utils/ocv_common.hpp>

struct BaseConfig {
    BaseConfig(const std::string& path_to_model) :
        pathToModel(path_to_model) {}

    std::string pathToModel;
    int numRequests = 1;
    ov::Core core;
    std::string deviceName;
};

class BaseModel {
public:
    BaseModel(const BaseConfig& config) : mConfig(config) {};
    virtual ~BaseModel() = default;

    bool enabled() const {
        return !mConfig.pathToModel.empty();
    }
protected:
    BaseConfig mConfig;
    std::string inputTensorName;
    std::vector<std::string> outputTensorsNames;

    virtual void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) = 0;
};

class AsyncModel : public BaseModel {
public:
    AsyncModel(const BaseConfig& config) :
        BaseModel(config) {
            if (!enabled()) {
                return;
            }
            slog::info << "Reading model: " << mConfig.pathToModel << slog::endl;
            std::shared_ptr<ov::Model> model = mConfig.core.read_model(mConfig.pathToModel);
            logBasicModelInfo(model);
            prepareInputsOutputs(model);
            ov::CompiledModel compiledModel = mConfig.core.compile_model(model, mConfig.deviceName);
            inferQueue.reset(new AsyncInferQueue(compiledModel, mConfig.numRequests));
            logCompiledModelInfo(compiledModel, mConfig.pathToModel, mConfig.deviceName);
        };

    std::vector<cv::Mat> infer(const std::vector<cv::Mat>& images);

private:
    cv::Size netInputSize;
    cv::Size origImageSize;
    std::unique_ptr<AsyncInferQueue> inferQueue;

    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
};

void alignFaces(std::vector<cv::Mat>& face_images,
                std::vector<cv::Mat>& landmarks_vec);

struct FaceBox {
    cv::Rect face;
    float confidence;
    explicit FaceBox(const cv::Rect& rect = cv::Rect(), float confidence = -1.0f) :
        face(rect), confidence(confidence) {}
};

struct DetectorConfig : public BaseConfig {
    DetectorConfig(const std::string& path_to_model) :
        BaseConfig(path_to_model) {}

    float confidenceThreshold = 0.5f;
    float increaseScaleX = 1.15f;
    float increaseScaleY = 1.15f;
    cv::Size inputSize = cv::Size(600, 600);
};

class FaceDetector : public BaseModel {
public:
    FaceDetector(const DetectorConfig& config) :
        BaseModel(config), mConfig(config) {
            slog::info << "Reading model: " << mConfig.pathToModel << slog::endl;
            std::shared_ptr<ov::Model> model = mConfig.core.read_model(mConfig.pathToModel);
            logBasicModelInfo(model);
            prepareInputsOutputs(model);
            ov::CompiledModel compiledModel = mConfig.core.compile_model(model, mConfig.deviceName);
            mRequest = std::make_shared<ov::InferRequest>(compiledModel.create_infer_request());
            logCompiledModelInfo(compiledModel, mConfig.pathToModel, mConfig.deviceName);
        };

    void submitData(const cv::Mat& inputImage);
    std::vector<FaceBox> getResults();

private:
    DetectorConfig mConfig;
    cv::Size netInputSize;
    cv::Size origImageSize;
    std::shared_ptr<ov::InferRequest> mRequest;
    size_t maxDetectionCount = 0;
    size_t detectedObjectSize = 0;
    static constexpr int emptyDetectionIndicator = -1;

    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
};
