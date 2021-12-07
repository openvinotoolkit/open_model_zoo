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
#include "models/image_model.h"

class LandmarksModel : public ImageModel {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param useAutoResize - if true, image will be resized by IE.
    /// @param postprocessType key for model model with heatmap output and simple output.
    LandmarksModel(const std::string& modelFileName, bool useAutoResize);

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult);

protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;
    std::unique_ptr<ResultBase> simplePostprocess(InferenceResult& infResult);
    std::unique_ptr<ResultBase> heatmapPostprocess(InferenceResult& infResult);
    std::vector<cv::Mat> split(float* data, const InferenceEngine::SizeVector& shape);
    std::vector<cv::Point2f> getMaxPreds(std::vector<cv::Mat> heatMaps);
    int sign(float number);
    cv::Mat affineTransform(cv::Point2f center, cv::Point2f scale,
        float rot, size_t dst_w, size_t dst_h, cv::Point2f shift, bool inv);
    cv::Point2f rotatePoint(cv::Point2f, float);
    cv::Point2f get3rdPoint(cv::Point2f a, cv::Point2f b);
    size_t numberLandmarks;
};
