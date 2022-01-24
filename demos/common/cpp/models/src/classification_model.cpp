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

#include <openvino/openvino.hpp>
#include <ngraph/ngraph.hpp>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "models/classification_model.h"

ClassificationModel::ClassificationModel(const std::string& modelFileName, size_t nTop, bool useAutoResize, const std::vector<std::string>& labels) :
    ImageModel(modelFileName, useAutoResize),
    nTop(nTop),
    labels(labels) {
}

std::unique_ptr<ResultBase> ClassificationModel::postprocess(InferenceResult& infResult) {
    ov::runtime::Tensor scoresTensor = infResult.outputsData.find(outputsNames[0])->second;
    const float* scoresPtr = scoresTensor.data<float>();
    ov::runtime::Tensor indicesTensor = infResult.outputsData.find(outputsNames[1])->second;
    const int* indicesPtr = indicesTensor.data<int>();

    ClassificationResult* result = new ClassificationResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    result->topLabels.reserve(scoresTensor.get_size());
    for (size_t i = 0; i < scoresTensor.get_size(); ++i) {
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

void ClassificationModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    const ov::OutputVector& inputsInfo = model->inputs();
    if (inputsInfo.size() != 1) {
        throw std::runtime_error("Classification model wrapper supports topologies only with 1 input");
    }
    inputsNames.push_back(model->input().get_any_name());

    const ov::Shape& inputShape = model->input().get_shape();
    if (inputShape.size() != 4 || inputShape[1] != 3) {
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");
    }
    if (inputShape[2] != inputShape[3]) {
        throw std::logic_error("Model input has incorrect image shape. Must be NxN square."
            " Got " + std::to_string(inputShape[2]) +
            "x" + std::to_string(inputShape[3]) + ".");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    if (useAutoResize) {
        ppp.input().tensor().
            set_element_type(ov::element::u8).
            set_spatial_dynamic_shape().
            set_layout({ "NHWC" });

        ppp.input().preprocess().
            convert_element_type(ov::element::f32).
            //convert_layout("NHWC").
            resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }
    else {
        ppp.input().tensor().
            set_element_type(ov::element::u8).
            set_layout({ "NHWC" });
    }

    ppp.input().model().set_layout("NCHW");

    // --------------------------- Prepare output blobs -----------------------------------------------------
    const ov::OutputVector& outputsInfo = model->outputs();
    if (outputsInfo.size() != 1) {
        throw std::runtime_error("Classification model wrapper supports topologies only with 1 output");
    }

    const ov::Shape& outputShape = model->output().get_shape();
    if (outputShape.size() != 2 && outputShape.size() != 4) {
        throw std::runtime_error("Classification model wrapper supports topologies only with 2-dimensional or 4-dimensional output");
    }
    if (outputShape.size() == 4 && (outputShape[2] != 1 || outputShape[3] != 1)) {
        throw std::runtime_error("Classification model wrapper supports topologies only with 4-dimensional output which has last two dimensions of size 1");
    }
    if (nTop > outputShape[1]) {
        throw std::runtime_error("The model provides " + std::to_string(outputShape[1]) + " classes, but " + std::to_string(nTop) + " labels are requested to be predicted");
    }

    if (outputShape[1] == labels.size() + 1) {
        labels.insert(labels.begin(), "other");
        slog::warn << "\tInserted 'other' label as first." << slog::endl;
    }
    else if (outputShape[1] != labels.size()) {
        throw std::logic_error("Model's number of classes and parsed labels must match (" + std::to_string(outputShape[1]) + " and " + std::to_string(labels.size()) + ')');
    }
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    // --------------------------- Adding softmax and topK output blobs ---------------------------
    auto nodes = model->get_ops();
    auto softmaxNodeIt = std::find_if(std::begin(nodes), std::end(nodes),
        [](const std::shared_ptr<ngraph::Node>& op) { return std::string(op->get_type_name()) == "Softmax"; });

    std::shared_ptr<ngraph::Node> softmaxNode;
    if (softmaxNodeIt == nodes.end()) {
        auto logitsNode = model->get_output_op(0)->input(0).get_source_output().get_node();
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
        std::make_shared<ngraph::Function>(res, model->get_parameters(), "classification");

    model = f;
    // manually set output tensors name for created topK node
    model->outputs()[0].set_names({"indices"});
    outputsNames.push_back("indices");
    model->outputs()[1].set_names({"scores"});
    outputsNames.push_back("scores");
}
