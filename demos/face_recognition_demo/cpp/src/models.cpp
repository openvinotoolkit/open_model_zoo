// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#include "models.hpp"
#include "reid_gallery.hpp"

#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <vector>

#include "utils/ocv_common.hpp"
#include "utils/image_utils.h"

void AsyncModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    if (model->inputs().size() != 1) {
        throw std::logic_error("Face landmarks/reidentification network should have only 1 input");
    }

    if (model->outputs().size() != 1) {
        throw std::logic_error("Face landmarks/reidentification network should have only 1 output");
    }
    inputTensorName = model->input().get_any_name();
    ov::OutputVector outputs = model->outputs();
    for (auto& item : outputs) {
        const std::string name = item.get_any_name();
        outputTensorsNames.push_back(name);
    }

    ov::Shape inputDims = model->input().get_shape();

    ov::Layout modelLayout = ov::layout::get_layout(model->input());
    if (modelLayout.empty()) {
        modelLayout = {"NCHW"};
    }
    netInputSize = cv::Size(inputDims[ov::layout::width_idx(modelLayout)],
                            inputDims[ov::layout::height_idx(modelLayout)]);

    ov::Shape outputDims = model->output().get_shape();

    ov::preprocess::PrePostProcessor ppp(model);
    ov::Layout desiredLayout = {"NHWC"};

    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout(desiredLayout);
    ppp.input().preprocess()
        .convert_layout(modelLayout)
        .convert_element_type(ov::element::f32);
    ppp.input().model().set_layout(modelLayout);

    model = ppp.build();
}

std::vector<cv::Mat> AsyncModel::infer(const std::vector<cv::Mat>& rois) {
    if (!enabled()) {
        return std::vector<cv::Mat>();
    }
    std::unordered_map<std::string, cv::Mat> input;
    for (size_t id = 0; id < rois.size(); ++id) {
        cv::Mat resizedImg = resizeImageExt(rois[id], netInputSize.width, netInputSize.height);
        input[inputTensorName] = std::move(resizedImg);
        inferQueue->submitData(input, id);
    }
    std::unordered_map<int64_t, std::map<std::string, ov::Tensor>> results = std::move(inferQueue->getResults());

    // create cv::Mats from results
    std::vector<cv::Mat> mats;
    for (size_t id = 0; id < rois.size(); ++id) {
        ov::Tensor tensor = results[id][outputTensorsNames[0]];
        if (tensor) {
            ov::Shape shape = tensor.get_shape();
            std::vector<int> tensorSizes(shape.size(), 0);
            for (size_t i = 0; i < tensorSizes.size(); ++i) {
                tensorSizes[i] = shape[i];
            }
            cv::Mat outTensor(tensorSizes, CV_32F, tensor.data<float>());
            outTensor = outTensor.reshape(1, outTensor.size[1]);
            mats.push_back(outTensor.clone());
        }
    }
    return mats;
}

namespace {
    cv::Rect truncateToValidRect(const cv::Rect& rect, const cv::Size& size) {
        auto tl = rect.tl(), br = rect.br();
        tl.x = std::max(0, std::min(size.width - 1, tl.x));
        tl.y = std::max(0, std::min(size.height - 1, tl.y));
        br.x = std::max(0, std::min(size.width, br.x));
        br.y = std::max(0, std::min(size.height, br.y));
        int w = std::max(0, br.x - tl.x);
        int h = std::max(0, br.y - tl.y);
        return cv::Rect(tl.x, tl.y, w, h);
    }

    cv::Rect increaseRect(const cv::Rect& r, float coeff_x, float coeff_y) {
        cv::Point2f tl = r.tl();
        cv::Point2f br = r.br();
        cv::Point2f c = (tl * 0.5f) + (br * 0.5f);
        cv::Point2f diff = c - tl;
        cv::Point2f newDiff{diff.x * coeff_x, diff.y * coeff_y};
        cv::Point2f newTl = c - newDiff;
        cv::Point2f newBr = c + newDiff;

        cv::Point newTlInt {static_cast<int>(std::floor(newTl.x)), static_cast<int>(std::floor(newTl.y))};
        cv::Point newBrInt {static_cast<int>(std::ceil(newBr.x)), static_cast<int>(std::ceil(newBr.y))};

        return cv::Rect(newTlInt, newBrInt);
    }
}  // namespace

void FaceDetector::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    if (model->inputs().size() != 1) {
        throw std::logic_error("Face Detection network should have only one input");
    }

    if (model->outputs().size() != 1) {
        throw std::logic_error("Face Detection network should have only one output");
    }
    inputTensorName = model->input().get_any_name();
    outputTensorsNames.push_back(model->output().get_any_name());

    if (mConfig.inputSize.area()) {
        model->reshape(ov::Shape({1, 3, static_cast<size_t>(mConfig.inputSize.height), static_cast<size_t>(mConfig.inputSize.width)}));
    }

    ov::Shape inputDims = model->input().get_shape();

    ov::Layout modelLayout = ov::layout::get_layout(model->input());
    if (modelLayout.empty()) {
        modelLayout = {"NCHW"};
    }
    netInputSize = cv::Size(inputDims[ov::layout::width_idx(modelLayout)],
                            inputDims[ov::layout::height_idx(modelLayout)]);

    ov::Shape outputDims = model->output().get_shape();
    maxDetectionCount = outputDims[2];
    detectedObjectSize = outputDims[3];
    if (detectedObjectSize != 7) {
        throw std::runtime_error("Face Detection network output layer should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
        throw std::runtime_error("Face Detection network output should have 4 dimensions, but had " +
            std::to_string(outputDims.size()));
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ov::Layout desiredLayout = {"NHWC"};

    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout(desiredLayout);
    ppp.input().preprocess()
        .convert_layout(modelLayout)
        .convert_element_type(ov::element::f32);
    ppp.input().model().set_layout(modelLayout);

    model = ppp.build();
}

// Function to start inference
void FaceDetector::submitData(const cv::Mat& inputImage) {
    origImageSize = inputImage.size();
    ov::Tensor inputTensor = mRequest->get_input_tensor();

    resize2tensor(inputImage, inputTensor);
    mRequest->start_async();
}

std::vector<FaceBox> FaceDetector::getResults() {
    mRequest->wait();
    const float* data = mRequest->get_output_tensor().data<float>();

    std::vector<FaceBox> detectedFaces;

    for (size_t det_id = 0; det_id < maxDetectionCount; ++det_id) {
        const int start_pos = det_id * detectedObjectSize;

        const float batchID = data[start_pos];
        if (batchID == emptyDetectionIndicator) {
            break;
        }

        const float score = std::min(std::max(0.0f, data[start_pos + 2]), 1.0f);
        if (score < mConfig.confidenceThreshold) {
            continue;
        }

        const float x0 = std::min(std::max(0.0f, data[start_pos + 3]), 1.0f) * origImageSize.width;
        const float y0 = std::min(std::max(0.0f, data[start_pos + 4]), 1.0f) * origImageSize.height;
        const float x1 = std::min(std::max(0.0f, data[start_pos + 5]), 1.0f) * origImageSize.width;
        const float y1 = std::min(std::max(0.0f, data[start_pos + 6]), 1.0f) * origImageSize.height;

        FaceBox detectedObject;
        detectedObject.confidence = score;
        detectedObject.face = cv::Rect(cv::Point(static_cast<int>(round(static_cast<double>(x0))),
                                         static_cast<int>(round(static_cast<double>(y0)))),
                               cv::Point(static_cast<int>(round(static_cast<double>(x1))),
                                         static_cast<int>(round(static_cast<double>(y1)))));


        detectedObject.face = truncateToValidRect(increaseRect(detectedObject.face,
                                                               mConfig.increaseScaleX,
                                                               mConfig.increaseScaleY),
                                                   cv::Size(static_cast<int>(origImageSize.width),
                                                            static_cast<int>(origImageSize.height)));

        if (detectedObject.face.area() > 0) {
            detectedFaces.emplace_back(detectedObject);
        }
    }

    return detectedFaces;
}
