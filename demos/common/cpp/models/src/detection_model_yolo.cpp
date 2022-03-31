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

#include "models/detection_model_yolo.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/slog.hpp>

#include "models/internal_model_data.h"
#include "models/results.h"

std::vector<float> defaultAnchors[] = {
    // YOLOv1v2
    {0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f, 5.47434f, 7.88282f, 3.52778f, 9.77052f, 9.16828f},
    // YOLOv3
    {10.0f,
     13.0f,
     16.0f,
     30.0f,
     33.0f,
     23.0f,
     30.0f,
     61.0f,
     62.0f,
     45.0f,
     59.0f,
     119.0f,
     116.0f,
     90.0f,
     156.0f,
     198.0f,
     373.0f,
     326.0f},
    // YOLOv4
    {12.0f,
     16.0f,
     19.0f,
     36.0f,
     40.0f,
     28.0f,
     36.0f,
     75.0f,
     76.0f,
     55.0f,
     72.0f,
     146.0f,
     142.0f,
     110.0f,
     192.0f,
     243.0f,
     459.0f,
     401.0f},
    // YOLOv4_Tiny
    {10.0f, 14.0f, 23.0f, 27.0f, 37.0f, 58.0f, 81.0f, 82.0f, 135.0f, 169.0f, 344.0f, 319.0f},
    // YOLOF
    {16.0f, 16.0f, 32.0f, 32.0f, 64.0f, 64.0f, 128.0f, 128.0f, 256.0f, 256.0f, 512.0f, 512.0f}};

const std::vector<int64_t> defaultMasks[] = {
    // YOLOv1v2
    {},
    // YOLOv3
    {},
    // YOLOv4
    {0, 1, 2, 3, 4, 5, 6, 7, 8},
    // YOLOv4_Tiny
    {1, 2, 3, 3, 4, 5},
    // YOLOF
    {0, 1, 2, 3, 4, 5}};

static inline float sigmoid(float x) {
    return 1.f / (1.f + exp(-x));
}

static inline float linear(float x) {
    return x;
}

ModelYolo::ModelYolo(const std::string& modelFileName,
                     float confidenceThreshold,
                     bool useAutoResize,
                     bool useAdvancedPostprocessing,
                     float boxIOUThreshold,
                     const std::vector<std::string>& labels,
                     const std::vector<float>& anchors,
                     const std::vector<int64_t>& masks,
                     const std::string& layout)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels, layout),
      boxIOUThreshold(boxIOUThreshold),
      useAdvancedPostprocessing(useAdvancedPostprocessing),
      yoloVersion(YOLO_V3),
      presetAnchors(anchors),
      presetMasks(masks) {}

void ModelYolo::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("YOLO model wrapper accepts models that have only 1 input");
    }

    const auto& input = model->input();
    const ov::Shape& inputShape = model->input().get_shape();
    ov::Layout inputLayout = getInputLayout(input);

    if (inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout({"NHWC"});

    if (useAutoResize) {
        ppp.input().tensor().set_spatial_dynamic_shape();

        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }

    ppp.input().model().set_layout(inputLayout);

    //--- Reading image input parameters
    inputsNames.push_back(model->input().get_any_name());
    netInputWidth = inputShape[ov::layout::width_idx(inputLayout)];
    netInputHeight = inputShape[ov::layout::height_idx(inputLayout)];

    // --------------------------- Prepare output  -----------------------------------------------------
    const ov::OutputVector& outputs = model->outputs();
    std::map<std::string, ov::Shape> outShapes;
    for (auto& out : outputs) {
        ppp.output(out.get_any_name()).tensor().set_element_type(ov::element::f32);
        if (out.get_shape().size() == 4) {
            if (out.get_shape()[ov::layout::height_idx(yoloRegionLayout)] !=
                    out.get_shape()[ov::layout::width_idx(yoloRegionLayout)] &&
                out.get_shape()[ov::layout::height_idx({"NHWC"})] == out.get_shape()[ov::layout::width_idx({"NHWC"})]) {
                yoloRegionLayout = {"NHWC"};
            }
            ppp.output(out.get_any_name()).tensor().set_layout(yoloRegionLayout);
        }
        outputsNames.push_back(out.get_any_name());
        outShapes[out.get_any_name()] = out.get_shape();
    }
    model = ppp.build();

    yoloVersion = YOLO_V3;
    bool isRegionFound = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (std::string("RegionYolo") == op->get_type_name()) {
            auto regionYolo = std::dynamic_pointer_cast<ov::op::v0::RegionYolo>(op);

            if (regionYolo) {
                if (!regionYolo->get_mask().size()) {
                    yoloVersion = YOLO_V1V2;
                }

                const auto& opName = op->get_friendly_name();
                for (const auto& out : outputs) {
                    if (out.get_node()->get_friendly_name() == opName ||
                        out.get_node()->get_input_node_ptr(0)->get_friendly_name() == opName) {
                        isRegionFound = true;
                        regions.emplace(out.get_any_name(), Region(regionYolo));
                    }
                }
            }
        }
    }

    if (!isRegionFound) {
        switch (outputsNames.size()) {
            case 1:
                yoloVersion = YOLOF;
                break;
            case 2:
                yoloVersion = YOLO_V4_TINY;
                break;
            case 3:
                yoloVersion = YOLO_V4;
                break;
        }

        int num = yoloVersion == YOLOF ? 6 : 3;
        isObjConf = yoloVersion == YOLOF ? 0 : 1;
        int i = 0;

        auto chosenMasks = presetMasks.size() ? presetMasks : defaultMasks[yoloVersion];
        if (chosenMasks.size() != num * outputs.size()) {
            throw std::runtime_error(std::string("Invalid size of masks array, got ") +
                                     std::to_string(presetMasks.size()) + ", should be " +
                                     std::to_string(num * outputs.size()));
        }

        std::sort(outputsNames.begin(),
                  outputsNames.end(),
                  [&outShapes, this](const std::string& x, const std::string& y) {
                      return outShapes[x][ov::layout::height_idx(yoloRegionLayout)] >
                             outShapes[y][ov::layout::height_idx(yoloRegionLayout)];
                  });

        for (const auto& name : outputsNames) {
            const auto& shape = outShapes[name];
            if (shape[ov::layout::channels_idx(yoloRegionLayout)] % num != 0) {
                throw std::logic_error(std::string("Output tenosor ") + name + " has wrong 2nd dimension");
            }
            regions.emplace(
                name,
                Region(shape[ov::layout::channels_idx(yoloRegionLayout)] / num - 4 - (isObjConf ? 1 : 0),
                       4,
                       presetAnchors.size() ? presetAnchors : defaultAnchors[yoloVersion],
                       std::vector<int64_t>(chosenMasks.begin() + i * num, chosenMasks.begin() + (i + 1) * num),
                       shape[ov::layout::width_idx(yoloRegionLayout)],
                       shape[ov::layout::height_idx(yoloRegionLayout)]));
            i++;
        }
    } else {
        // Currently externally set anchors and masks are supported only for YoloV4
        if (presetAnchors.size() || presetMasks.size()) {
            slog::warn << "Preset anchors and mask can be set for YoloV4 model only. "
                          "This model is not YoloV4, so these options will be ignored."
                       << slog::endl;
        }
    }
}

std::unique_ptr<ResultBase> ModelYolo::postprocess(InferenceResult& infResult) {
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    std::vector<DetectedObject> objects;

    // Parsing outputs
    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();

    for (auto& output : infResult.outputsData) {
        this->parseYOLOOutput(output.first,
                              output.second,
                              netInputHeight,
                              netInputWidth,
                              internalData.inputImgHeight,
                              internalData.inputImgWidth,
                              objects);
    }

    if (useAdvancedPostprocessing) {
        // Advanced postprocessing
        // Checking IOU threshold conformance
        // For every i-th object we're finding all objects it intersects with, and comparing confidence
        // If i-th object has greater confidence than all others, we include it into result
        for (const auto& obj1 : objects) {
            bool isGoodResult = true;
            for (const auto& obj2 : objects) {
                if (obj1.labelID == obj2.labelID && obj1.confidence < obj2.confidence &&
                    intersectionOverUnion(obj1, obj2) >= boxIOUThreshold) {  // if obj1 is the same as obj2, condition
                                                                             // expression will evaluate to false anyway
                    isGoodResult = false;
                    break;
                }
            }
            if (isGoodResult) {
                result->objects.push_back(obj1);
            }
        }
    } else {
        // Classic postprocessing
        std::sort(objects.begin(), objects.end(), [](const DetectedObject& x, const DetectedObject& y) {
            return x.confidence > y.confidence;
        });
        for (size_t i = 0; i < objects.size(); ++i) {
            if (objects[i].confidence == 0)
                continue;
            for (size_t j = i + 1; j < objects.size(); ++j)
                if (intersectionOverUnion(objects[i], objects[j]) >= boxIOUThreshold)
                    objects[j].confidence = 0;
            result->objects.push_back(objects[i]);
        }
    }

    return std::unique_ptr<ResultBase>(result);
}

void ModelYolo::parseYOLOOutput(const std::string& output_name,
                                const ov::Tensor& tensor,
                                const unsigned long resized_im_h,
                                const unsigned long resized_im_w,
                                const unsigned long original_im_h,
                                const unsigned long original_im_w,
                                std::vector<DetectedObject>& objects) {
    // --------------------------- Extracting layer parameters -------------------------------------
    auto it = regions.find(output_name);
    if (it == regions.end()) {
        throw std::runtime_error(std::string("Can't find output layer with name ") + output_name);
    }
    auto& region = it->second;

    int sideW = 0;
    int sideH = 0;
    unsigned long scaleH;
    unsigned long scaleW;
    switch (yoloVersion) {
        case YOLO_V1V2:
            sideH = region.outputHeight;
            sideW = region.outputWidth;
            scaleW = region.outputWidth;
            scaleH = region.outputHeight;
            break;
        case YOLO_V3:
        case YOLO_V4:
        case YOLO_V4_TINY:
        case YOLOF:
            sideH = static_cast<int>(tensor.get_shape()[ov::layout::height_idx(yoloRegionLayout)]);
            sideW = static_cast<int>(tensor.get_shape()[ov::layout::width_idx(yoloRegionLayout)]);
            scaleW = resized_im_w;
            scaleH = resized_im_h;
            break;
    }

    auto entriesNum = sideW * sideH;
    const float* outData = tensor.data<float>();

    auto postprocessRawData =
        (yoloVersion == YOLO_V4 || yoloVersion == YOLO_V4_TINY || yoloVersion == YOLOF) ? sigmoid : linear;

    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < entriesNum; ++i) {
        int row = i / sideW;
        int col = i % sideW;
        for (int n = 0; n < region.num; ++n) {
            //--- Getting region data
            int obj_index = calculateEntryIndex(entriesNum,
                                                region.coords,
                                                region.classes + isObjConf,
                                                n * entriesNum + i,
                                                region.coords);
            int box_index =
                calculateEntryIndex(entriesNum, region.coords, region.classes + isObjConf, n * entriesNum + i, 0);
            float scale = isObjConf ? postprocessRawData(outData[obj_index]) : 1;

            //--- Preliminary check for confidence threshold conformance
            if (scale >= confidenceThreshold) {
                //--- Calculating scaled region's coordinates
                float x, y;
                if (yoloVersion == YOLOF) {
                    x = (static_cast<float>(col) / sideW +
                         outData[box_index + 0 * entriesNum] * region.anchors[2 * n] / scaleW) *
                        original_im_w;
                    y = (static_cast<float>(row) / sideH +
                         outData[box_index + 1 * entriesNum] * region.anchors[2 * n + 1] / scaleH) *
                        original_im_h;
                } else {
                    x = static_cast<float>((col + postprocessRawData(outData[box_index + 0 * entriesNum])) / sideW *
                                           original_im_w);
                    y = static_cast<float>((row + postprocessRawData(outData[box_index + 1 * entriesNum])) / sideH *
                                           original_im_h);
                }
                float height = static_cast<float>(std::exp(outData[box_index + 3 * entriesNum]) *
                                                  region.anchors[2 * n + 1] * original_im_h / scaleH);
                float width = static_cast<float>(std::exp(outData[box_index + 2 * entriesNum]) * region.anchors[2 * n] *
                                                 original_im_w / scaleW);

                DetectedObject obj;
                obj.x = clamp(x - width / 2, 0.f, static_cast<float>(original_im_w));
                obj.y = clamp(y - height / 2, 0.f, static_cast<float>(original_im_h));
                obj.width = clamp(width, 0.f, static_cast<float>(original_im_w - obj.x));
                obj.height = clamp(height, 0.f, static_cast<float>(original_im_h - obj.y));

                for (size_t j = 0; j < region.classes; ++j) {
                    int class_index = calculateEntryIndex(entriesNum,
                                                          region.coords,
                                                          region.classes + isObjConf,
                                                          n * entriesNum + i,
                                                          region.coords + isObjConf + j);
                    float prob = scale * postprocessRawData(outData[class_index]);

                    //--- Checking confidence threshold conformance and adding region to the list
                    if (prob >= confidenceThreshold) {
                        obj.confidence = prob;
                        obj.labelID = j;
                        obj.label = getLabelName(obj.labelID);
                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

int ModelYolo::calculateEntryIndex(int totalCells, int lcoords, size_t lclasses, int location, int entry) {
    int n = location / totalCells;
    int loc = location % totalCells;
    return (n * (lcoords + lclasses) + entry) * totalCells + loc;
}

double ModelYolo::intersectionOverUnion(const DetectedObject& o1, const DetectedObject& o2) {
    double overlappingWidth = fmin(o1.x + o1.width, o2.x + o2.width) - fmax(o1.x, o2.x);
    double overlappingHeight = fmin(o1.y + o1.height, o2.y + o2.height) - fmax(o1.y, o2.y);
    double intersectionArea =
        (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
    double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
    return intersectionArea / unionArea;
}

ModelYolo::Region::Region(const std::shared_ptr<ov::op::v0::RegionYolo>& regionYolo) {
    coords = regionYolo->get_num_coords();
    classes = regionYolo->get_num_classes();
    auto mask = regionYolo->get_mask();
    num = mask.size();

    auto shape = regionYolo->get_input_shape(0);
    outputWidth = shape[3];
    outputHeight = shape[2];

    if (num) {
        // Parsing YoloV3 parameters
        anchors.resize(num * 2);

        for (int i = 0; i < num; ++i) {
            anchors[i * 2] = regionYolo->get_anchors()[mask[i] * 2];
            anchors[i * 2 + 1] = regionYolo->get_anchors()[mask[i] * 2 + 1];
        }
    } else {
        // Parsing YoloV2 parameters
        num = regionYolo->get_num_regions();
        anchors = regionYolo->get_anchors();
        if (anchors.empty()) {
            anchors = defaultAnchors[YOLO_V1V2];
            num = 5;
        }
    }
}

ModelYolo::Region::Region(size_t classes,
                          int coords,
                          const std::vector<float>& anchors,
                          const std::vector<int64_t>& masks,
                          size_t outputWidth,
                          size_t outputHeight)
    : classes(classes),
      coords(coords),
      outputWidth(outputWidth),
      outputHeight(outputHeight) {
    num = masks.size();

    if (anchors.size() == 0 || anchors.size() % 2 != 0) {
        throw std::runtime_error("Explicitly initialized region should have non-empty even-sized regions vector");
    }

    if (num) {
        this->anchors.resize(num * 2);

        for (int i = 0; i < num; ++i) {
            this->anchors[i * 2] = anchors[masks[i] * 2];
            this->anchors[i * 2 + 1] = anchors[masks[i] * 2 + 1];
        }
    } else {
        this->anchors = anchors;
        num = anchors.size() / 2;
    }
}
