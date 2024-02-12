/*
// Copyright (C) 2022-2024 Intel Corporation
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
#include <memory>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "models/detection_model.h"

class ModelYoloX: public DetectionModel {
public:
    /// Constructor.
    /// @param modelFileName name of model to load
    /// @param confidenceThreshold - threshold to eliminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param boxIOUThreshold - threshold to treat separate output regions as one object for filtering
    /// during postprocessing (only one of them should stay). The default value is 0.5
    /// @param labels - array of labels for every class. If this array is empty or contains less elements
    /// than actual classes number, default "Label #N" will be shown for missing items.
    /// @param layout - model input layout
    ModelYoloX(const std::string& modelFileName,
                    float confidenceThreshold,
                    float boxIOUThreshold = 0.5,
                    const std::vector<std::string>& labels = std::vector<std::string>(),
                    const std::string& layout = "");

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;
    std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, ov::InferRequest& request) override;

protected:
    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
    void setStridesGrids();

    float boxIOUThreshold;
    std::vector<std::pair<size_t, size_t>> grids;
    std::vector<size_t> expandedStrides;
    static const size_t numberOfClasses = 80;
};
