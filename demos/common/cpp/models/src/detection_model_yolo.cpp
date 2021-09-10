/*
// Copyright (C) 2018-2020 Intel Corporation
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
#include <utils/slog.hpp>
#include <utils/common.hpp>
#include <ngraph/ngraph.hpp>

std::vector<float> defaultAnchors[] = {
    // YOLOv1v2
    { 0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f, 5.47434f, 7.88282f, 3.52778f, 9.77052f, 9.16828f },
    // YOLOv3
    { 10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f,
      30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f,
      116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f},
    // YOLOv4
    { 12.0f, 16.0f, 19.0f, 36.0f, 40.0f, 28.0f,
      36.0f, 75.0f, 76.0f, 55.0f, 72.0f, 146.0f,
      142.0f, 110.0f, 192.0f, 243.0f, 459.0f, 401.0f},
    // YOLOv4_Tiny
    { 10.0f, 14.0f, 23.0f, 27.0f, 37.0f, 58.0f,
      81.0f, 82.0f, 135.0f, 169.0f, 344.0f, 319.0f}
};

const std::vector<int64_t> defaultMasks[] = {
    // YOLOv1v2
    {},
    // YOLOv3
    {},
    // YOLOv4
    {0, 1, 2, 3, 4, 5, 6, 7, 8 },
    // YOLOv4_Tiny
    {1, 2, 3, 3, 4, 5}
};

static inline float sigmoid(float x) {
    return 1.f / (1.f + exp(-x));
}
static inline float linear(float x) {
    return x;
}


ModelYolo::ModelYolo(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize,
    bool useAdvancedPostprocessing, float boxIOUThreshold, const std::vector<std::string>& labels,
    const std::vector<float>& anchors, const std::vector<int64_t>& masks) :
    DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels),
    boxIOUThreshold(boxIOUThreshold),
    useAdvancedPostprocessing(useAdvancedPostprocessing),
    yoloVersion(YOLO_V3),
    presetAnchors(anchors),
    presetMasks(masks) {
}

void ModelYolo::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }

    InferenceEngine::InputInfo::Ptr& input = inputInfo.begin()->second;
    inputsNames.push_back(inputInfo.begin()->first);
    input->setPrecision(InferenceEngine::Precision::U8);
    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(InferenceEngine::Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    }

    //--- Reading image input parameters
    const InferenceEngine::TensorDesc& inputDesc = inputInfo.begin()->second->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        if (output.second->getDims().size() == 4) {
            output.second->setLayout(InferenceEngine::Layout::NCHW);
        }
        outputsNames.push_back(output.first);
    }

    yoloVersion = YOLO_V3;
    bool isRegionFound = false;
    if (auto ngraphFunction = (cnnNetwork).getFunction()) {
        for (const auto op : ngraphFunction->get_ops()) {
            auto outputLayer = outputInfo.find(op->get_friendly_name());
            if (outputLayer != outputInfo.end()) {
                auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);

                if (regionYolo) {
                    isRegionFound = true;

                    if (!regionYolo->get_mask().size()) {
                        yoloVersion = YOLO_V1V2;
                    }

                    regions.emplace(outputLayer->first, Region(regionYolo));
                }
            }
        }
    }
    else {
        throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
    }

    if(!isRegionFound)
    {
        yoloVersion = outputsNames.size() == 2 ? YOLO_V4_TINY : YOLO_V4;

        int num = 3;
        int i = 0;

        auto chosenMasks = presetMasks.size() ? presetMasks : defaultMasks[yoloVersion];
        if(chosenMasks.size() != num * outputInfo.size()) {
            throw std::runtime_error(std::string("Invalid size of masks array, got ") + std::to_string(presetMasks.size()) +
                ", should be " + std::to_string(num * outputInfo.size()));
        }

        std::sort(outputsNames.begin(), outputsNames.end(),
            [&outputInfo](const std::string& x, const std::string&  y) {return outputInfo[x]->getDims()[2] > outputInfo[y]->getDims()[2];});

        for (const auto& name : outputsNames) {
            auto& output = outputInfo[name];
            auto shape = output->getDims();
            auto classes = shape[1] / num - 5;
            if (shape[1] % num != 0) {
                throw std::runtime_error(std::string("The output blob ") + name + " has wrong 2nd dimension");
            }
            regions.emplace(name, Region(classes, 4,
                presetAnchors.size() ? presetAnchors : defaultAnchors[yoloVersion],
                std::vector<int64_t>(chosenMasks.begin() + i*num, chosenMasks.begin() + (i+1)*num),
                shape[3], shape[2]));
            i++;
        }
    }
    else {
        // Currently externally set anchors and masks are supported only for YoloV4
        if(presetAnchors.size() || presetMasks.size()){
            slog::warn << "Preset anchors and mask can be set for YoloV4 model only. "
                "This model is not YoloV4, so these options will be ignored." << slog::endl;
        }
    }
}

std::unique_ptr<ResultBase> ModelYolo::postprocess(InferenceResult & infResult) {
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    std::vector<DetectedObject> objects;

    // Parsing outputs
    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();

    for (auto& output : infResult.outputsData) {
        this->parseYOLOOutput(output.first, output.second, netInputHeight, netInputWidth,
            internalData.inputImgHeight, internalData.inputImgWidth, objects);
    }

    if (useAdvancedPostprocessing) {
        // Advanced postprocessing
        // Checking IOU threshold conformance
        // For every i-th object we're finding all objects it intersects with, and comparing confidence
        // If i-th object has greater confidence than all others, we include it into result
        for (const auto& obj1 : objects) {
            bool isGoodResult = true;
            for (const auto& obj2 : objects) {
                if (obj1.labelID == obj2.labelID && obj1.confidence < obj2.confidence && intersectionOverUnion(obj1, obj2) >= boxIOUThreshold) { // if obj1 is the same as obj2, condition expression will evaluate to false anyway
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
        std::sort(objects.begin(), objects.end(), [](const DetectedObject& x, const DetectedObject& y) { return x.confidence > y.confidence; });
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
    const InferenceEngine::Blob::Ptr& blob, const unsigned long resized_im_h,
    const unsigned long resized_im_w, const unsigned long original_im_h,
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
    switch(yoloVersion) {
    case YOLO_V1V2:
        sideH = region.outputHeight;
        sideW = region.outputWidth;
        scaleW = region.outputWidth;
        scaleH = region.outputHeight;
        break;
    case YOLO_V3:
    case YOLO_V4:
    case YOLO_V4_TINY:
        sideH = static_cast<int>(blob->getTensorDesc().getDims()[2]);
        sideW = static_cast<int>(blob->getTensorDesc().getDims()[3]);
        scaleW = resized_im_w;
        scaleH = resized_im_h;
        break;
    }

    auto entriesNum = sideW * sideH;
    const float* output_blob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    auto postprocessRawData = (yoloVersion == YOLO_V4 || yoloVersion == YOLO_V4_TINY) ? sigmoid : linear;

    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < entriesNum; ++i) {
        int row = i / sideW;
        int col = i % sideW;
        for (int n = 0; n < region.num; ++n) {
            //--- Getting region data from blob
            int obj_index = calculateEntryIndex(entriesNum, region.coords, region.classes, n * entriesNum + i, region.coords);
            int box_index = calculateEntryIndex(entriesNum, region.coords, region.classes, n * entriesNum + i, 0);
            float scale = postprocessRawData(output_blob[obj_index]);

            //--- Preliminary check for confidence threshold conformance
            if (scale >= confidenceThreshold){
                //--- Calculating scaled region's coordinates
                double x = (col + postprocessRawData(output_blob[box_index + 0 * entriesNum])) / sideW * original_im_w;
                double y = (row + postprocessRawData(output_blob[box_index + 1 * entriesNum])) / sideH * original_im_h;
                double height = std::exp(output_blob[box_index + 3 * entriesNum]) * region.anchors[2 * n + 1] * original_im_h / scaleH;
                double width = std::exp(output_blob[box_index + 2 * entriesNum]) * region.anchors[2 * n] * original_im_w / scaleW;

                DetectedObject obj;
                obj.x = (float)std::max((x-width/2), 0.);
                obj.y = (float)std::max((y-height/2), 0.);
                obj.width = std::min((float)width, original_im_w - obj.x);
                obj.height = std::min((float)height, original_im_h - obj.y);

                for (int j = 0; j < region.classes; ++j) {
                    int class_index = calculateEntryIndex(entriesNum, region.coords, region.classes, n * entriesNum + i, region.coords + 1 + j);
                    float prob = scale * postprocessRawData(output_blob[class_index]);

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

int ModelYolo::calculateEntryIndex(int totalCells, int lcoords, int lclasses, int location, int entry) {
    int n = location / totalCells;
    int loc = location % totalCells;
    return (n * (lcoords + lclasses + 1) + entry) * totalCells + loc;
}

double ModelYolo::intersectionOverUnion(const DetectedObject& o1, const DetectedObject& o2) {
    double overlappingWidth = fmin(o1.x + o1.width, o2.x + o2.width) - fmax(o1.x, o2.x);
    double overlappingHeight = fmin(o1.y + o1.height, o2.y + o2.height) - fmax(o1.y, o2.y);
    double intersectionArea = (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
    double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
    return intersectionArea / unionArea;
}

ModelYolo::Region::Region(const std::shared_ptr<ngraph::op::RegionYolo>& regionYolo) {
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

ModelYolo::Region::Region(int classes, int coords, const std::vector<float>& anchors, const std::vector<int64_t>& masks, int outputWidth, int outputHeight) :
    classes(classes), coords(coords),
    outputWidth(outputWidth), outputHeight(outputHeight) {
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
    }
    else {
        this->anchors = anchors;
        num = anchors.size() / 2;
    }
}
