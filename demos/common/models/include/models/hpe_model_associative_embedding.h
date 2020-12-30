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
    /// @param confidenceThreshold - threshold to eleminate low-confidence keypoints.
    /// Any keypoint with confidence lower than this threshold will be ignored.
    HpeAssociativeEmbedding(const std::string& modelFileName, double aspectRatio, int targetSize, float confidenceThreshold);

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;

    static const int numJoints = 17;

protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork) override;

    cv::Size inputLayerSize;
    cv::Size resizedImageSize;
    double aspectRatio;
    float confidenceThreshold;
    int targetSize;

    const cv::Vec3f meanPixel = cv::Vec3f::all(128);
    const int stride = 32;
    const int maxNumPeople = 30;
    const float detectionThreshold = 0.1f;
    const float tagThreshold = 1.0f;
    const bool useDetectionVal = true;
    const bool doAdjust = true;
    const bool doRefine = true;
    const bool ignoreTooMuch = true;
    const float delta = 0.0f;

    void reshape(InferenceEngine::CNNNetwork & cnnNetwork) override;

    void convertTo3D(std::vector<cv::Mat>& , float* data, const InferenceEngine::SizeVector& shape);

    std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& aembdsMaps,
                                        const std::vector<cv::Mat>& nmsHeatMaps) const;
};
