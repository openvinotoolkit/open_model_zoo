/*
// Copyright (C) 2021-2022 Intel Corporation
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

#include "models/model_base.h"

namespace ov {
class InferRequest;
}  // namespace ov
struct InputData;
struct InternalModelData;

class ImageModel : public ModelBase {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param useAutoResize - if true, image is resized by openvino
    /// @param layout - model input layout
    ImageModel(const std::string& modelFileName, bool useAutoResize, const std::string& layout = "");

    std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, ov::InferRequest& request) override;

protected:
    bool useAutoResize;

    size_t netInputHeight = 0;
    size_t netInputWidth = 0;
};
