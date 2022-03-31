/*
// Copyright (C) 2021-2022 Intel Corporation
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
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <utils/image_utils.h>

#include "models/image_model.h"

namespace ov {
class InferRequest;
class Model;
class Shape;
}  // namespace ov
struct HumanPose;
struct InferenceResult;
struct InputData;
struct InternalModelData;
struct ResultBase;

class HpeAssociativeEmbedding : public ImageModel {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param aspectRatio - the ratio of input width to its height.
    /// @param targetSize - the length of a short image side used for model reshaping.
    /// @param confidenceThreshold - threshold to eliminate low-confidence poses.
    /// Any pose with confidence lower than this threshold will be ignored.
    /// @param layout - model input layout
    HpeAssociativeEmbedding(const std::string& modelFileName,
                            double aspectRatio,
                            int targetSize,
                            float confidenceThreshold,
                            const std::string& layout = "",
                            float delta = 0.0,
                            RESIZE_MODE resizeMode = RESIZE_KEEP_ASPECT);

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, ov::InferRequest& request) override;

protected:
    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;

    cv::Size inputLayerSize;
    double aspectRatio;
    int targetSize;
    float confidenceThreshold;
    float delta;
    RESIZE_MODE resizeMode;

    std::string embeddingsTensorName;
    std::string heatmapsTensorName;
    std::string nmsHeatmapsTensorName;

    static const int numJoints = 17;
    static const int stride = 32;
    static const int maxNumPeople = 30;
    static const cv::Vec3f meanPixel;
    static const float detectionThreshold;
    static const float tagThreshold;

    void changeInputSize(std::shared_ptr<ov::Model>& model);

    std::string findTensorByName(const std::string& tensorName, const std::vector<std::string>& outputsNames);

    std::vector<cv::Mat> split(float* data, const ov::Shape& shape);

    std::vector<HumanPose> extractPoses(std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& aembdsMaps,
                                        const std::vector<cv::Mat>& nmsHeatMaps) const;
};
