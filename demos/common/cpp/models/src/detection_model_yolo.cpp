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

using namespace InferenceEngine;

ModelYolo3::ModelYolo3(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize,
    bool useAdvancedPostprocessing, float boxIOUThreshold, const std::vector<std::string>& labels) :
    DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels),
    boxIOUThreshold(boxIOUThreshold),
    useAdvancedPostprocessing(useAdvancedPostprocessing) {
}

void ModelYolo3::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }

    InputInfo::Ptr& input = inputInfo.begin()->second;
    inputsNames.push_back(inputInfo.begin()->first);
    input->setPrecision(Precision::U8);
    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(Layout::NCHW);
    }

    //--- Reading image input parameters
    const TensorDesc& inputDesc = inputInfo.begin()->second->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto& output : outputInfo) {
        output.second->setPrecision(Precision::FP32);
        output.second->setLayout(Layout::NCHW);
        outputsNames.push_back(output.first);
    }

    if (auto ngraphFunction = (cnnNetwork).getFunction()) {
        for (const auto op : ngraphFunction->get_ops()) {
            auto outputLayer = outputInfo.find(op->get_friendly_name());
            if (outputLayer != outputInfo.end()) {
                auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                if (!regionYolo) {
                    throw std::runtime_error("Invalid output type: " +
                        std::string(op->get_type_info().name) + ". RegionYolo expected");
                }

                regions.emplace(outputLayer->first, Region(regionYolo));
            }
        }
    }
    else {
        throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
    }
}

std::unique_ptr<ResultBase> ModelYolo3::postprocess(InferenceResult & infResult) {
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    std::vector<DetectedObject> objects;

    // Parsing outputs
    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();

    for (auto& output : infResult.outputsData) {
        this->parseYOLOV3Output(output.first, output.second, netInputHeight, netInputWidth,
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

void ModelYolo3::parseYOLOV3Output(const std::string& output_name,
    const InferenceEngine::Blob::Ptr& blob, const unsigned long resized_im_h,
    const unsigned long resized_im_w, const unsigned long original_im_h,
    const unsigned long original_im_w,
    std::vector<DetectedObject>& objects) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w) {
        throw std::runtime_error("Invalid size of output " + output_name +
            " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
            ", current W = " + std::to_string(out_blob_h));
    }

    // --------------------------- Extracting layer parameters -------------------------------------
    auto it = regions.find(output_name);
    if(it == regions.end()) {
        throw std::runtime_error(std::string("Can't find output layer with name ") + output_name);
    }
    auto& region = it->second;

    auto side = out_blob_h;
    auto side_square = side * side;
    const float* output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < region.num; ++n) {
            //--- Getting region data from blob
            int obj_index = calculateEntryIndex(side, region.coords, region.classes, n * side * side + i, region.coords);
            int box_index = calculateEntryIndex(side, region.coords, region.classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];

            //--- Preliminary check for confidence threshold conformance
            if (scale >= confidenceThreshold){
                //--- Calculating scaled region's coordinates
                double x = (col + output_blob[box_index + 0 * side_square]) / side * original_im_w;
                double y = (row + output_blob[box_index + 1 * side_square]) / side * original_im_h;
                double height = std::exp(output_blob[box_index + 3 * side_square]) * region.anchors[2 * n + 1] * original_im_h / resized_im_h;
                double width = std::exp(output_blob[box_index + 2 * side_square]) * region.anchors[2 * n] * original_im_w / resized_im_w;

                DetectedObject obj;
                obj.x = (float)(x-width/2);
                obj.y = (float)(y-height/2);
                obj.width = (float)(width);
                obj.height = (float)(height);

                for (int j = 0; j < region.classes; ++j) {
                    int class_index = calculateEntryIndex(side, region.coords, region.classes, n * side_square + i, region.coords + 1 + j);
                    float prob = scale * output_blob[class_index];

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

int ModelYolo3::calculateEntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

double ModelYolo3::intersectionOverUnion(const DetectedObject& o1, const DetectedObject& o2) {
    double overlappingWidth = fmin(o1.x + o1.width, o2.x + o2.width) - fmax(o1.x, o2.x);
    double overlappingHeight = fmin(o1.y + o1.height, o2.y + o2.height) - fmax(o1.y, o2.y);
    double intersectionArea = (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
    double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
    return intersectionArea / unionArea;
}

ModelYolo3::Region::Region(const std::shared_ptr<ngraph::op::RegionYolo>& regionYolo) {
    coords = regionYolo->get_num_coords();
    classes = regionYolo->get_num_classes();
    auto mask = regionYolo->get_mask();
    num = mask.size();
    anchors.resize(num * 2);

    for (int i = 0; i < num; ++i) {
        anchors[i * 2] = regionYolo->get_anchors()[mask[i] * 2];
        anchors[i * 2 + 1] = regionYolo->get_anchors()[mask[i] * 2 + 1];
    }
}
