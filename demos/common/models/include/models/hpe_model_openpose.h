/*
// Copyright (C) 2018-2020 Intel Corporation
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
#include "human_pose.h"

class HPEOpenPose : public ModelBase {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    HPEOpenPose(const std::string& modelFileName);

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;

    static const size_t keypointsNumber = 18;

protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork) override;

    const int minJointsNumber = 3;
    const int stride = 8;
    cv::Vec4i pad = cv::Vec4i::all(0);
    const cv::Vec3f meanPixel = cv::Vec3f::all(128);
    const float minPeaksDistance = 3.0f;
    const float midPointsScoreThreshold = 0.05f;
    const float foundMidPointsRatioThreshold = 0.8f;
    const float minSubsetScore = 0.2f;
    const int upsampleRatio = 4;
    cv::Size inputLayerSize;

    std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& pafs) const;
    void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;

    cv::Size reshape(InferenceEngine::CNNNetwork & cnnNetwork, int targetSize=0) override;
};
