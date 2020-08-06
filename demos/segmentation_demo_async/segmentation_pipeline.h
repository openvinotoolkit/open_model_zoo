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
class SegmentationPipeline :
    public PipelineBase
{
public:
    struct SegmentationResult
    {
        int64_t frameId = -1;
        cv::Mat mask;

        bool IsEmpty() const { return frameId < 0; }
        SegmentationResult(){}
        SegmentationResult(int64_t frameId, cv::Mat mask){
            this->frameId = frameId;
            this->mask = mask;
        }
    };

public:
    SegmentationPipeline();
    virtual ~SegmentationPipeline();

    virtual void PrepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork);

    int64_t submitImage(cv::Mat img);
    SegmentationResult getSegmentationResult();
protected:
    const cv::Vec3b& class2Color(int classId);

    std::string imageInputName;
    int outHeight = 0;
    int outWidth = 0;
    int outChannels = 0;

    std::vector<cv::Vec3b> colors;
    std::mt19937 rng;
    std::uniform_int_distribution<int> distr;
};

