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
#include <opencv2/core.hpp>
#include <inference_engine.hpp>
#include <map>
#include "internal_model_data.h"

struct MetaData;
struct ResultBase {
    virtual ~ResultBase() {}

    int64_t frameId = -1;

    std::shared_ptr<MetaData> metaData;
    bool IsEmpty() { return frameId < 0; }

    template<class T> T& asRef() {
        return dynamic_cast<T&>(*this);
    }

    template<class T> const T& asRef() const {
        return dynamic_cast<const T&>(*this);
    }
};

struct InferenceResult : public ResultBase {
    std::shared_ptr<InternalModelData> internalModelData;
    std::map<std::string, InferenceEngine::MemoryBlob::Ptr> outputsData;

    /// Returns pointer to first output blob
    /// This function is a useful addition to direct access to outputs list as many models have only one output
    /// @returns pointer to first output blob
    InferenceEngine::MemoryBlob::Ptr getFirstOutputBlob() {
        if (outputsData.empty())
            throw std::out_of_range("Outputs map is empty.");
        return outputsData.begin()->second;
    }

    /// Returns true if object contains no valid data
    /// @returns true if object contains no valid data
    bool IsEmpty() { return outputsData.empty(); }
};

struct ClassificationResult : public ResultBase {
    std::vector<std::pair<unsigned int, std::string>> topLabels;
};

struct DetectedObject : public cv::Rect2f {
    unsigned int labelID;
    std::string label;
    float confidence;
};

struct DetectionResult : public ResultBase {
    std::vector<DetectedObject> objects;
};

struct RetinaFaceDetectionResult : public DetectionResult {
    std::vector<cv::Point2f> landmarks;
};

struct SegmentationResult : public ResultBase {
    cv::Mat mask;
};

struct HumanPose {
    std::vector<cv::Point2f> keypoints;
    float score;
};

struct HumanPoseResult : public ResultBase {
    std::vector<HumanPose> poses;
};
