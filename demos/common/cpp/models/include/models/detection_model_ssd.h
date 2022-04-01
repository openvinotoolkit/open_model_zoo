/*
// Copyright (C) 2020-2022 Intel Corporation
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
#include <stddef.h>

#include <memory>
#include <string>
#include <vector>

#include "models/detection_model.h"

namespace ov {
class InferRequest;
class Model;
}  // namespace ov
struct InferenceResult;
struct InputData;
struct InternalModelData;
struct ResultBase;

class ModelSSD : public DetectionModel {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param confidenceThreshold - threshold to eliminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by openvino.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    /// @param labels - array of labels for every class. If this array is empty or contains less elements
    /// than actual classes number, default "Label #N" will be shown for missing items.
    /// @param layout - model input layout
    ModelSSD(const std::string& modelFileName,
             float confidenceThreshold,
             bool useAutoResize,
             const std::vector<std::string>& labels = std::vector<std::string>(),
             const std::string& layout = "");

    std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, ov::InferRequest& request) override;
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    std::unique_ptr<ResultBase> postprocessSingleOutput(InferenceResult& infResult);
    std::unique_ptr<ResultBase> postprocessMultipleOutputs(InferenceResult& infResult);
    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
    void prepareSingleOutput(std::shared_ptr<ov::Model>& model);
    void prepareMultipleOutputs(std::shared_ptr<ov::Model>& model);
    size_t objectSize = 0;
    size_t detectionsNumId = 0;
};
