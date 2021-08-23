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
#include "models/model_base.h"
#include <utils/ocv_common.hpp>

class ImageModel : public ModelBase {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param useAutoResize - if true, image is resized by IE.
    //// @param inputTransform - if not trivial, it applies input normalization (means subtraction and/or division by scales per channel).
    ImageModel(const std::string& modelFileName, bool useAutoResize, InputTransform& inputTransform = InputTransform());

    virtual std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;

protected:
    bool useAutoResize;
    InputTransform inputTransform;

    size_t netInputHeight = 0;
    size_t netInputWidth = 0;
};
