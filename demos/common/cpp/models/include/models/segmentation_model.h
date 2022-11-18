/*
// Copyright (C) 2020-2022 Intel Corporation
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

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "models/image_model.h"

namespace ov {
class Model;
}  // namespace ov
struct InferenceResult;
struct ResultBase;

#pragma once
class SegmentationModel : public ImageModel {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// @param useAutoResize - if true, image will be resized by openvino.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    /// @param layout - model input layout
    SegmentationModel(const std::string& modelFileName, bool useAutoResize, const std::string& layout = "");

    static std::vector<std::string> loadLabels(const std::string& labelFilename);

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;

    int outHeight = 0;
    int outWidth = 0;
    int outChannels = 0;
};
