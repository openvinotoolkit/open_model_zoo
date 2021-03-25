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
#include "detection_model.h"
class ModelSSD : public DetectionModel {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param confidenceThreshold - threshold to eliminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by IE.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    /// @param labels - array of labels for every class. If this array is empty or contains less elements
    /// than actual classes number, default "Label #N" will be shown for missing items.
    ModelSSD(const std::string& modelFileName,
        float confidenceThreshold, bool useAutoResize,
        const std::vector<std::string>& labels = std::vector<std::string>());

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    std::unique_ptr<ResultBase> postprocessSingleOutput(InferenceResult& infResult);
    std::unique_ptr<ResultBase> postprocessMultipleOutputs(InferenceResult& infResult);
    void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;
    void prepareSingleOutput(InferenceEngine::OutputsDataMap& outputInfo);
    void prepareMultipleOutputs(InferenceEngine::OutputsDataMap& outputInfo);
    size_t maxProposalCount = 0;
    size_t objectSize = 0;
};
