/*
// Copyright (C) 2020-2021 Intel Corporation
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
#include <vector>
#include "detection_model.h"
#include <utils/nms.hpp>

class ModelRetinaFace
    : public DetectionModel {
public:
    struct Anchor {
        float left;
        float top;
        float right;
        float bottom;

        float getWidth() const { return (right - left) + 1.0f; }
        float getHeight() const { return (bottom - top) + 1.0f; }
        float getXCenter() const { return left + (getWidth() - 1.0f) / 2.0f; }
        float getYCenter() const { return top + (getHeight() - 1.0f) / 2.0f; }
    };

    static const int LANDMARKS_NUM = 5;
    static const int INIT_VECTOR_SIZE = 200;
    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    /// @param confidenceThreshold - threshold to eliminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by IE.
    /// @param labels - array of labels for every class. If this array is empty or contains less elements
    /// than actual classes number, default "Label #N" will be shown for missing items.
    ModelRetinaFace(const std::string& model_name, float confidenceThreshold, bool useAutoResize, float boxIOUThreshold);
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

    enum EOutputType {
        OT_BBOX,
        OT_SCORES,
        OT_LANDMARK,
        OT_MASKSCORES,
        OT_MAX
    };

    std::vector <std::string> separateOutputsNames[OT_MAX];
    const std::vector<AnchorCfgLine> anchorCfg;
    std::map<int, std::vector <Anchor>> anchorsFpn;
    std::vector<std::vector<Anchor>> anchors;

    void generateAnchorsFpn();
    void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;
};
