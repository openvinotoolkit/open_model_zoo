/*
// Copyright (C) 2020-2021 Intel Corporation
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
#include <vector>
#include "detection_model.h"
#include <utils/nms.hpp>
#include <string>

class ModelRetinaFacePT : public DetectionModel {
public:
    struct Box {
        float cX;
        float cY;
        float width;
        float height;
    };

    struct Rect {
        float left;
        float top;
        float right;
        float bottom;

        float getWidth() const { return (right - left) + 1.0f; }
        float getHeight() const { return (bottom - top) + 1.0f; }
        float getXCenter() const { return left + (getWidth() - 1.0f) / 2.0f; }
        float getYCenter() const { return top + (getHeight() - 1.0f) / 2.0f; }
    };

    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    /// @param confidenceThreshold - threshold to eliminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by IE.
    /// @param boxIOUThreshold - threshold for NMS boxes filtering, varies in [0.0, 1.0] range.
    ModelRetinaFacePT(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize, float boxIOUThreshold);
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    size_t landmarksNum;
    const float boxIOUThreshold;
    float variance[2] = { 0.1f, 0.2f };

    enum EOutputType {
        OT_BBOX,
        OT_SCORES,
        OT_LANDMARK,
        OT_MAX
    };

    std::vector<ModelRetinaFacePT::Box> priors;

    std::vector<size_t> filterByScore(const InferenceEngine::MemoryBlob::Ptr& rawData, const float confidenceThreshold);
    std::vector<float> getFilteredScores(const InferenceEngine::MemoryBlob::Ptr& rawData, const std::vector<size_t>& indicies);
    std::vector<cv::Point2f> getFilteredLandmarks(const InferenceEngine::MemoryBlob::Ptr& rawData,
        const std::vector<size_t>& indicies, int imgWidth, int imgHeight);
    std::vector<ModelRetinaFacePT::Box> generatePriorData();
    std::vector<ModelRetinaFacePT::Rect> getFilteredProposals(const InferenceEngine::MemoryBlob::Ptr& rawData,
        const std::vector<size_t>& indicies, int imgWidth, int imgHeight);

    void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;
};
