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
#include <opencv2/core.hpp>

using namespace InferenceEngine;

ModelYolo3::ModelYolo3(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize,
    bool useAdvancedPostprocessing, float boxIOUThreshold, const std::vector<std::string>& labels, const std::string& regionsFile) :
    DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels),
    boxIOUThreshold(boxIOUThreshold),
    useAdvancedPostprocessing(useAdvancedPostprocessing), regionsFile(regionsFile) {
}

template<class InputsDataMap, class OutputsDataMap>
void ModelYolo3::checkInputsOutputs(const InputsDataMap& inputInfo, const OutputsDataMap& outputInfo) {
    // --------------------------- Check input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts YOLO networks that have only one input");
    }
    const auto& input = inputInfo.begin()->second;
    if (input->getPrecision() != InferenceEngine::Precision::U8) {
        throw std::logic_error("This demo accepts networks with U8 input precision");
    }

    // -------------------Reading image input parameters----------------------------------------------------
    inputsNames.push_back(inputInfo.begin()->first);
    const TensorDesc& inputDesc = inputInfo.begin()->second->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Check output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    for (const auto& output : outputInfo) {
        outputsNames.push_back(output.first);
        if (output.second->getPrecision() != InferenceEngine::Precision::FP32) {
            throw std::logic_error("This demo accepts networks with FP32 output precision");
        }
    }
}

void ModelYolo3::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -----------------------------------------------
    const auto& inputInfo = cnnNetwork.getInputsInfo();
    const auto& outputInfo = cnnNetwork.getOutputsInfo();
    for (const auto& input : inputInfo) {
        if (input.second->getTensorDesc().getDims().size() == 4) {
            if (useAutoResize) {
                input.second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
                input.second->getInputData()->setLayout(InferenceEngine::Layout::NHWC);
            }
            else {
                input.second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
            }
            input.second->setPrecision(InferenceEngine::Precision::U8);
        }
        else if (input.second->getTensorDesc().getDims().size() == 2) {  // 2nd input contains image info
            input.second->setPrecision(InferenceEngine::Precision::FP32);
        }
    }

    for (const auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::NCHW);
    }

    // --------------------------- Check input & output ----------------------------------------------------
    checkInputsOutputs(inputInfo, outputInfo);

    //---------------------------- Read yolo regions from IR -----------------------------------------------
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

void ModelYolo3::checkCompiledNetworkInputsOutputs() {
    checkInputsOutputs(execNetwork.GetInputsInfo(), execNetwork.GetOutputsInfo());

    //------------------------- Read yolo regions from file -----------------------------------------------
    cv::FileStorage fs(regionsFile, cv::FileStorage::READ);
    cv::FileNode regionsYolo = fs["Regions"];
    cv::FileNodeIterator it = regionsYolo.begin(), endIt = regionsYolo.end();

    for (; it != endIt; ++it) {
        std::vector<float> anchors;
        (*it)["anchors"] >> anchors;
        regions.emplace(static_cast<std::string>((*it)["name"]), Region((int)(*it)["num"], (int)(*it)["classes"], (int)(*it)["coords"], anchors));
    }
    fs.release();
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
