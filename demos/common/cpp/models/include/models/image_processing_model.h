/*
// Copyright (C) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writingb  software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "image_processing_model.h"

#pragma once

class ImageProcessingModel : public ModelBase{
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param useAutoResize - if true, image will be resized by IE.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    ImageProcessingModel(const std::string& modelFileName, const cv::Size inputImageShape=cv::Size(0, 0));

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;
    void reshape(InferenceEngine::CNNNetwork & cnnNetwork) override;
    cv::Size getViewInfo() { return viewInfo; }
protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork) override;
    cv::Size inputSize;
    int outHeight = 0;
    int outWidth = 0;
    int outChannels = 0;
    std::string type;
    cv::Size viewInfo;
};
