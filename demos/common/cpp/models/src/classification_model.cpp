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

#include <ngraph/ngraph.hpp>
#include "models/classification_model.h"
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

ClassificationModel::ClassificationModel(const std::string& modelFileName, size_t nTop, bool useAutoResize, const std::vector<std::string>& labels) :
    ImageModel(modelFileName, useAutoResize),
    nTop(nTop),
    labels(labels) {
}

std::unique_ptr<ResultBase> ClassificationModel::postprocess(InferenceResult& infResult) {
    InferenceEngine::MemoryBlob::Ptr scoresBlob = infResult.outputsData.find(outputsNames[0])->second;
    const float* scoresPtr = scoresBlob->rmap().as<float*>();
    InferenceEngine::MemoryBlob::Ptr indicesBlob = infResult.outputsData.find(outputsNames[1])->second;
    const int* indicesPtr = indicesBlob->rmap().as<int*>();

    ClassificationResult* result = new ClassificationResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    result->topLabels.reserve(scoresBlob->size());
    for (int i = 0; i < scoresBlob->size(); ++i) {
        result->topLabels.emplace_back(indicesPtr[i], labels[indicesPtr[i]], scoresPtr[i]);
    }

    return retVal;
}

std::vector<std::string> ClassificationModel::loadLabels(const std::string& labelFilename) {
    std::vector<std::string> labels;

    /* Read labels */
    std::ifstream inputFile(labelFilename);
    if (!inputFile.is_open())
        throw std::runtime_error("Can't open the labels file: " + labelFilename);
    std::string labelsLine;
    while (std::getline(inputFile, labelsLine)) {
        size_t labelBeginIdx = labelsLine.find(' ');
        size_t labelEndIdx = labelsLine.find(',');  // can be npos when class has only one label
        if (labelBeginIdx == std::string::npos) {
            throw std::runtime_error("The labels file has incorrect format.");
        }
        labels.push_back(labelsLine.substr(labelBeginIdx + 1, labelEndIdx - (labelBeginIdx + 1)));
    }
    if (labels.empty())
        throw std::logic_error("File is empty: " + labelFilename);

    return labels;
}

void ClassificationModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 input");
    inputsNames.push_back(inputShapes.begin()->first);
    InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
    if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");
    if (inSizeVector[2] != inSizeVector[3])
        throw std::logic_error("Model input has incorrect image shape. Must be NxN square."
                                " Got " + std::to_string(inSizeVector[2]) +
                                "x" + std::to_string(inSizeVector[3]) + ".");

    InferenceEngine::InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(InferenceEngine::Precision::U8);
    if (useAutoResize) {
        inputInfo.getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        inputInfo.getInputData()->setLayout(InferenceEngine::Layout::NHWC);
    }
    else
        inputInfo.getInputData()->setLayout(InferenceEngine::Layout::NCHW);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    const InferenceEngine::OutputsDataMap& outputsDataMap = cnnNetwork.getOutputsInfo();
    if (outputsDataMap.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");
    InferenceEngine::Data& data = *outputsDataMap.begin()->second;
    const InferenceEngine::SizeVector& outSizeVector = data.getTensorDesc().getDims();
    if (outSizeVector.size() != 2 && outSizeVector.size() != 4)
        throw std::runtime_error("Demo supports topologies only with 2-dimensional or 4-dimensional output");
    if (outSizeVector.size() == 4 && (outSizeVector[2] != 1 || outSizeVector[3] != 1))
        throw std::runtime_error("Demo supports topologies only with 4-dimensional output which has last two dimensions of size 1");
    if (nTop > outSizeVector[1])
        throw std::runtime_error("The model provides " + std::to_string(outSizeVector[1]) + " classes, but " + std::to_string(nTop) + " labels are requested to be predicted");
    if (outSizeVector[1] == labels.size() + 1) {
        labels.insert(labels.begin(), "other");
        slog::warn << "\tInserted 'other' label as first." << slog::endl;
    }
    else if (outSizeVector[1] != labels.size())
        throw std::logic_error("Model's number of classes and parsed labels must match (" + std::to_string(outSizeVector[1]) + " and " + std::to_string(labels.size()) + ')');

    data.setPrecision(InferenceEngine::Precision::FP32);

    // --------------------------- Adding softmax and topK output blobs ---------------------------
    if (auto ngraphFunction = (cnnNetwork).getFunction()) {
        auto nodes = ngraphFunction->get_ops();
        auto softmaxNodeIt = std::find_if(std::begin(nodes), std::end(nodes),
            [](auto op) { return std::string(op->get_type_name()) == "Softmax"; });

        std::shared_ptr<ngraph::Node> softmaxNode;
        if (softmaxNodeIt == nodes.end()) {
            auto logitsNode = ngraphFunction->get_output_op(0)->input(0).get_source_output().get_node();
            softmaxNode = std::make_shared<ngraph::op::v1::Softmax>(logitsNode->output(0), 1);
        }
        else {
            softmaxNode = *softmaxNodeIt;
        }
        const auto k = std::make_shared<ngraph::op::Constant>(ngraph::element::i32, ngraph::Shape{}, std::vector<size_t>{nTop});
        ngraph::op::v1::TopK::Mode mode = ngraph::op::v1::TopK::Mode::MAX;
        ngraph::op::v1::TopK::SortType sort = ngraph::op::v1::TopK::SortType::SORT_VALUES;
        std::shared_ptr<ngraph::Node> topkNode = std::make_shared<ngraph::op::v1::TopK>(softmaxNode, k, 1, mode, sort);

        auto scores = std::make_shared<ngraph::op::Result>(topkNode->output(0));
        auto indices = std::make_shared<ngraph::op::Result>(topkNode->output(1));
        std::vector<std::shared_ptr<ngraph::op::v0::Result>> res({ scores, indices });
        std::shared_ptr<ngraph::Function> f =
            std::make_shared<ngraph::Function>(res, ngraphFunction->get_parameters(), "classification");

        cnnNetwork = InferenceEngine::CNNNetwork(f);
        ngraphFunction = cnnNetwork.getFunction();

        for (auto& it : cnnNetwork.getOutputsInfo()) {
            outputsNames.push_back(it.first);
        }
        // outputsNames[0] - scores, outputsNames[1] - indices
        std::sort(outputsNames.begin(), outputsNames.end());
    }
    else {
        throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
    }

}
