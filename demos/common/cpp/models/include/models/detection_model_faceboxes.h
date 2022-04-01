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
#include <utility>
#include <vector>

#include "models/detection_model.h"

namespace ov {
class Model;
}  // namespace ov
struct InferenceResult;
struct ResultBase;

class ModelFaceBoxes : public DetectionModel {
public:
    struct Anchor {
        float left;
        float top;
        float right;
        float bottom;

        float getWidth() const {
            return (right - left) + 1.0f;
        }
        float getHeight() const {
            return (bottom - top) + 1.0f;
        }
        float getXCenter() const {
            return left + (getWidth() - 1.0f) / 2.0f;
        }
        float getYCenter() const {
            return top + (getHeight() - 1.0f) / 2.0f;
        }
    };
    static const int INIT_VECTOR_SIZE = 200;

    ModelFaceBoxes(const std::string& modelFileName,
                   float confidenceThreshold,
                   bool useAutoResize,
                   float boxIOUThreshold,
                   const std::string& layout = "");
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    size_t maxProposalsCount;
    const float boxIOUThreshold;
    const std::vector<float> variance;
    const std::vector<int> steps;
    const std::vector<std::vector<int>> minSizes;
    std::vector<Anchor> anchors;
    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
    void priorBoxes(const std::vector<std::pair<size_t, size_t>>& featureMaps);
};
