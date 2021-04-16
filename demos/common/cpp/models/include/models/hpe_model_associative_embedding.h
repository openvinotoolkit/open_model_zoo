/*
// Copyright (C) 2021 Intel Corporation
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
#include "model_base.h"

class HpeAssociativeEmbedding : public ModelBase {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param aspectRatio - the ratio of input width to its height.
    /// @param targetSize - the length of a short image side used for network reshaping.
    /// @param confidenceThreshold - threshold to eliminate low-confidence poses.
    /// Any pose with confidence lower than this threshold will be ignored.
    HpeAssociativeEmbedding(const std::string& modelFileName, double aspectRatio, int targetSize, float confidenceThreshold,
                            float delta = 0.0, std::string paddingMode = "right_bottom");

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;

protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;

    cv::Size inputLayerSize;
    double aspectRatio;
    int targetSize;
    float confidenceThreshold;
    float delta;
    std::string paddingMode;

    std::string embeddingsBlobName;
    std::string heatmapsBlobName;
    std::string nmsHeatmapsBlobName;

    static const int numJoints = 17;
    static const int stride = 32;
    static const int maxNumPeople = 30;
    static const cv::Vec3f meanPixel;
    static const float detectionThreshold;
    static const float tagThreshold;

    void changeInputSize(InferenceEngine::CNNNetwork& cnnNetwork);

    std::string findLayerByName(const std::string layerName,
                                const std::vector<std::string>& outputsNames);

    std::vector<cv::Mat> split(float* data, const InferenceEngine::SizeVector& shape);

    std::vector<HumanPose> extractPoses(std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& aembdsMaps,
                                        const std::vector<cv::Mat>& nmsHeatMaps) const;
};
