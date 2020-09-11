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

#include "pipeline_base.h"
#include "opencv2/core.hpp"
#pragma once
class DetectionPipeline :
    public PipelineBase
{
public:
    struct ObjectDesc : public cv::Rect2f {
        unsigned int labelID;
        std::string label;
        float confidence;
    };

    struct DetectionResult : public ResultBase {
        std::vector<ObjectDesc> objects;
    };

public:
    DetectionPipeline();
    virtual ~DetectionPipeline();

    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    virtual void init(const std::string& model_name, const CnnConfig& config,
        float confidenceThreshold, bool useAutoResize, InferenceEngine::Core* engine = nullptr);

    virtual void PrepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork);

    int64_t submitImage(cv::Mat img);
    DetectionResult getProcessedResult();

    cv::Mat obtainAndRenderData();

    void loadLabels(const std::string& labelFilename);
    std::vector<std::string> labels;

protected:
    std::string imageInfoInputName;
    size_t netInputHeight=0;
    size_t netInputWidth=0;

    bool useAutoResize=false;
    size_t maxProposalCount=0;
    size_t objectSize=0;
    float confidenceThreshold=0;
};

