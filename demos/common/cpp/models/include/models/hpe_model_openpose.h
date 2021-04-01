/*
// Copyright (C) 2018-2021 Intel Corporation
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
#include "image_model.h"

class HPEOpenPose : public ImageModel {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param aspectRatio - the ratio of input width to its height.
    /// @param targetSize - the height used for network reshaping.
    /// @param confidenceThreshold - threshold to eliminate low-confidence keypoints.
    HPEOpenPose(const std::string& modelFileName, double aspectRatio, int targetSize, float confidenceThreshold);

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;

    static const size_t keypointsNumber = 18;

protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;

    static const int minJointsNumber = 3;
    static const int stride = 8;
    static const int upsampleRatio = 4;
    static const cv::Vec3f meanPixel;
    static const float minPeaksDistance;
    static const float midPointsScoreThreshold;
    static const float foundMidPointsRatioThreshold;
    static const float minSubsetScore;
    cv::Size inputLayerSize;
    double aspectRatio;
    int targetSize;
    float confidenceThreshold;

    std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& pafs) const;
    void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;

    void changeInputSize(InferenceEngine::CNNNetwork& cnnNetwork);
};
