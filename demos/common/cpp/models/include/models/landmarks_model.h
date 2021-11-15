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
    LandmarksModel(const std::string& modelFileName, bool useAutoResize,std::string postprocessType);

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult);

protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;
    std::unique_ptr<ResultBase> LandmarksModel::simplePostprocess(InferenceResult& infResult);
    //std::unique_ptr<ResultBase> LandmarksModel::heatmapPostprocess(InferenceResult& infResult);
    size_t frameHeight, frameWidth;
    int numberLandmarks;
    std::string postprocessType;// simple or heatmap
};
