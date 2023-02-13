/*
// Copyright (C) 2020-2023 Intel Corporation
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

#include "models/detection_model.h"
#include "models/detection_model_centernet.h"
#include "models/detection_model_faceboxes.h"
#include "models/detection_model_retinaface.h"
#include "models/detection_model_retinaface_pt.h"
#include "models/detection_model_ssd.h"
#include "models/detection_model_yolo.h"
#include "models/detection_model_yolov3_onnx.h"
#include "models/detection_model_yolox.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/image_model.h"

DetectionModel::DetectionModel(const std::string& modelFileName,
                               float confidenceThreshold,
                               bool useAutoResize,
                               const std::vector<std::string>& labels,
                               const std::string& layout)
    : ImageModel(modelFileName, useAutoResize, layout),
      labels(labels),
      confidenceThreshold(confidenceThreshold) {}

std::unique_ptr<DetectionModel> DetectionModel::create_model(const std::string& modelFileName, std::shared_ptr<ov::Core> core, std::string model_type, float confidence_threshold, std::vector<std::string> labels) {
    if (!core) {
        core.reset(new ov::Core{});
    }
    std::shared_ptr<ov::Model> model = core->read_model(modelFileName);
    if (model_type.empty()) {
        model_type = model->get_rt_info<std::string>("model_info", "model_type");
    }
    if (-std::numeric_limits<float>::infinity() == confidence_threshold) {
        confidence_threshold = stof(model->get_rt_info<std::string>("model_info", "confidence_threshold"));
    }
    if (labels.empty()) {
        labels = split(model->get_rt_info<std::string>("model_info", "labels"), ' ');
    }
    std::string FLAGS_layout;
    bool FLAGS_auto_resize = false;
    float FLAGS_iou_t = 0.5;
    std::vector<float> anchors;
    std::vector<int64_t> masks;
    if (model_type == "centernet") {
        return std::unique_ptr<DetectionModel>(new ModelCenterNet(modelFileName, static_cast<float>(confidence_threshold), labels, FLAGS_layout));
    } else if (model_type == "faceboxes") {
        return std::unique_ptr<DetectionModel>(new ModelFaceBoxes(modelFileName,
                                        static_cast<float>(confidence_threshold),
                                        FLAGS_auto_resize,
                                        static_cast<float>(FLAGS_iou_t),
                                        FLAGS_layout));
    } else if (model_type == "retinaface") {
        return std::unique_ptr<DetectionModel>(new ModelRetinaFace(modelFileName,
                                        static_cast<float>(confidence_threshold),
                                        FLAGS_auto_resize,
                                        static_cast<float>(FLAGS_iou_t),
                                        FLAGS_layout));
    } else if (model_type == "retinaface-pytorch") {
        return std::unique_ptr<DetectionModel>(new ModelRetinaFacePT(modelFileName,
                                            static_cast<float>(confidence_threshold),
                                            FLAGS_auto_resize,
                                            static_cast<float>(FLAGS_iou_t),
                                            FLAGS_layout));
    } else if (model_type == "ssd") {
        return std::unique_ptr<DetectionModel>(new ModelSSD(modelFileName, static_cast<float>(confidence_threshold), FLAGS_auto_resize, labels, FLAGS_layout));
    } else if (model_type == "yolo") {
        bool FLAGS_yolo_af = true;  // Use advanced postprocessing/filtering algorithm for YOLO
        return std::unique_ptr<DetectionModel>(new ModelYolo(modelFileName,
                                    static_cast<float>(confidence_threshold),
                                    FLAGS_auto_resize,
                                    FLAGS_yolo_af,
                                    static_cast<float>(FLAGS_iou_t),
                                    labels,
                                    anchors,
                                    masks,
                                    FLAGS_layout));
    } else if (model_type == "yolov3-onnx") {
        return std::unique_ptr<DetectionModel>(new ModelYoloV3ONNX(modelFileName,
                                        static_cast<float>(confidence_threshold),
                                        labels,
                                        FLAGS_layout));
    } else if (model_type == "yolox") {
        return std::unique_ptr<DetectionModel>(new ModelYoloX(modelFileName,
                                    static_cast<float>(confidence_threshold),
                                    static_cast<float>(FLAGS_iou_t),
                                    labels,
                                    FLAGS_layout));
    } else {
        throw std::runtime_error{"No model type or invalid model type (-at) provided: " + model_type};
    }
}

std::vector<std::string> DetectionModel::loadLabels(const std::string& labelFilename) {
    std::vector<std::string> labelsList;

    /* Read labels (if any) */
    if (!labelFilename.empty()) {
        std::ifstream inputFile(labelFilename);
        if (!inputFile.is_open())
            throw std::runtime_error("Can't open the labels file: " + labelFilename);
        std::string label;
        while (std::getline(inputFile, label)) {
            labelsList.push_back(label);
        }
        if (labelsList.empty())
            throw std::logic_error("File is empty: " + labelFilename);
    }

    return labelsList;
}
