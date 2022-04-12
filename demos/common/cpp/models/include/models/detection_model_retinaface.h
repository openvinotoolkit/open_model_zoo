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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "models/detection_model.h"

namespace ov {
class Model;
}  // namespace ov
struct InferenceResult;
struct ResultBase;

class ModelRetinaFace : public DetectionModel {
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

    static const int LANDMARKS_NUM = 5;
    static const int INIT_VECTOR_SIZE = 200;
    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    /// @param confidenceThreshold - threshold to eliminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by openvino.
    /// @param boxIOUThreshold - threshold for NMS boxes filtering, varies in [0.0, 1.0] range.
    /// @param layout - model input layout
    ModelRetinaFace(const std::string& model_name,
                    float confidenceThreshold,
                    bool useAutoResize,
                    float boxIOUThreshold,
                    const std::string& layout = "");
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    struct AnchorCfgLine {
        int stride;
        std::vector<int> scales;
        int baseSize;
        std::vector<int> ratios;
    };

    bool shouldDetectMasks;
    bool shouldDetectLandmarks;
    const float boxIOUThreshold;
    const float maskThreshold;
    float landmarkStd;

    enum OutputType { OUT_BOXES, OUT_SCORES, OUT_LANDMARKS, OUT_MASKSCORES, OUT_MAX };

    std::vector<std::string> separateOutputsNames[OUT_MAX];
    const std::vector<AnchorCfgLine> anchorCfg;
    std::map<int, std::vector<Anchor>> anchorsFpn;
    std::vector<std::vector<Anchor>> anchors;

    void generateAnchorsFpn();
    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
};
