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
#include "detection_pipeline.h"
class DetectionPipelineSSD :
    public DetectionPipeline
{
public:
    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    /// @param cnnConfig - fine tuning configuration for CNN model
    /// @param confidenceThreshold - threshold to eleminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by IE.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    /// @param labels - array of labels for every class. If this array is empty or contains less elements
    /// than actual classes number, default "Label #N" will be shown for missing items.
    /// @param engine - pointer to InferenceEngine::Core instance to use.
    /// If it is omitted, new instance of InferenceEngine::Core will be created inside.
    virtual void init(const std::string& model_name, const CnnConfig& cnnConfig,
        float confidenceThreshold, bool useAutoResize,
        const std::vector<std::string>& labels = std::vector<std::string>(),
        InferenceEngine::Core* engine = nullptr);

    virtual DetectionResult getProcessedResult(bool shouldKeepOrder = true);
protected:
    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork);
};

