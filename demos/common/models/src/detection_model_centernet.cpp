/*
// Copyright (C) 2018-2021 Intel Corporation
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

#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/imgproc.hpp>

#include <ngraph/ngraph.hpp>
#include <samples/slog.hpp>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include "models/detection_model_centernet.h"

using namespace InferenceEngine;

ModelCenterNet::ModelCenterNet(const std::string& modelFileName,
    float confidenceThreshold, bool useAutoResize, float boxIOUThreshold, const std::vector<std::string>& labels)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels),
    boxIOUThreshold(boxIOUThreshold) {
}

void ModelCenterNet::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }

    InputInfo::Ptr& input = inputInfo.begin()->second;
    const TensorDesc& inputDesc = input->getTensorDesc();
    input->setPrecision(Precision::U8);

    if (inputDesc.getDims()[1] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }

    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(Layout::NCHW);
    }

    // --------------------------- Reading image input parameters -------------------------------------------
    std::string imageInputName = inputInfo.begin()->first;
    inputsNames.push_back(imageInputName);
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;

    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());

    if (outputInfo.size() != 3) {
        throw std::logic_error("This demo expect networks that have 3 outputs blobs");
    }

    const TensorDesc& outputDesc = outputInfo.begin()->second->getTensorDesc();
    maxProposalsCount = outputDesc.getDims()[1];

    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::CHW);
        outputsNames.push_back(output.first);
    }

}

cv::Point2f getDir(const cv::Point2f& srcPoint, float rotRadius) {
    float sn = std::sinf(rotRadius);
    float cs = std::cosf(rotRadius);

    cv::Point2f srcResult({ 0, 0 });
    srcResult.x = srcPoint.x * cs - srcPoint.y * sn;
    srcResult.y = srcPoint.x * sn + srcPoint.y * cs;

    return srcResult;
}

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f direct = a - b;
    return b + cv::Point2f({ -direct.y, direct.x });
}

cv::Mat getAffineTransform(float centerX, float centerY, float scale, float rot, int outputWidth, int outputHeight, bool inv = false) {
    //if not isinstance(scale, np.ndarray) and not isinstance(scale, list) :
    //    scale = np.array([scale, scale], dtype = np.float32)

    //    scale_tmp = scale
    //    src_w = scale_tmp[0]
    //    dst_w, dst_h = output_size

    //    rot_rad = np.pi * rot / 180
    //    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    //    dst_dir = np.array([0, dst_w * -0.5], dtype = np.float32)

    //    dst = np.zeros((3, 2), dtype = np.float32)
    //    src = np.zeros((3, 2), dtype = np.float32)
    //    src[0, :], src[1, :] = center, center + src_dir
    //    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    //    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    //    src[2:, : ] = get_3rd_point(src[0, :], src[1, :])
    //    dst[2:, : ] = get_3rd_point(dst[0, :], dst[1, :])

    //    if inv:
    //trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    //    else:
    //trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    //    return trans
    float srcW = scale;
    float rotRad =  M_PI * rot / 180.0f;
    auto srcDir = getDir({ 0, -0.5f * srcW }, rotRad);
    cv::Point2f dstDir({ 0,  -0.5f * outputWidth });
    std::vector<cv::Point2f> src(3, { 0 , 0 });
    std::vector<cv::Point2f> dst(3, { 0 , 0 });

    src[0] = { centerX, centerY };
    src[1] = srcDir + src[0];
    src[2] = get3rdPoint(src[0], src[1]);

    dst[0] = { outputWidth * 0.5f, outputHeight * 0.5f };
    dst[1] = dst[0] + dstDir;
    dst[2] = get3rdPoint(dst[0], dst[1]);

    cv::Mat trans;
    if (inv) {
        trans = cv::getAffineTransform(src, dst);
    }
    else {
        trans = cv::getAffineTransform(src, dst);
    }

    return trans;
}

std::shared_ptr<InternalModelData> ModelCenterNet::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {

    //height, width = image.shape[0:2]
    //    center = np.array([width / 2., height / 2.], dtype = np.float32)
    //    scale = max(height, width)
    //    trans_input = self.get_affine_transform(center, scale, 0, [self.w, self.h])
    //    resized_image = cv2.warpAffine(image, trans_input, (self.w, self.h), flags = cv2.INTER_LINEAR)
    //    resized_image = np.transpose(resized_image, (2, 0, 1))
    auto& img = inputData.asRef<ImageInputData>().inputImage;

    int imgWidth = img.cols;
    int imgHeight = img.rows;

    float centerX = imgWidth / 2;
    float centerY = imgHeight / 2;

    int scale = std::max(imgWidth, imgHeight);
    auto transInput = getAffineTransform(centerX, centerY, scale, 0, imgWidth, imgHeight);
    cv::Mat resizedImg;
    cv::warpAffine(img, resizedImg, transInput, { netInputWidth, netInputHeight }, cv::INTER_LINEAR);

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(inputsNames[0], wrapMat2Blob(img));
    }
    else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
        matU8ToBlob<uint8_t>(img, frameBlob);
    }

    return std::shared_ptr<InternalModelData>(new InternalImageModelData(img.cols, img.rows));
}

std::unique_ptr<ResultBase> ModelCenterNet::postprocess(InferenceResult& infResult) {
        //heat = outputs[self._output_layer_names[0]][0]
        //reg = outputs[self._output_layer_names[1]][0]
        //wh = outputs[self._output_layer_names[2]][0]
        //heat = np.exp(heat) / (1 + np.exp(heat))
        //height, width = heat.shape[1:3]
        //num_predictions = 100

        //heat = self._nms(heat)
        //scores, inds, clses, ys, xs = self._topk(heat, K = num_predictions)
        //reg = self._tranpose_and_gather_feat(reg, inds)

        //reg = reg.reshape((num_predictions, 2))
        //xs = xs.reshape((num_predictions, 1)) + reg[:, 0 : 1]
        //ys = ys.reshape((num_predictions, 1)) + reg[:, 1 : 2]

        //wh = self._tranpose_and_gather_feat(wh, inds)
        //wh = wh.reshape((num_predictions, 2))
        //clses = clses.reshape((num_predictions, 1))
        //scores = scores.reshape((num_predictions, 1))
        //bboxes = np.concatenate((xs - wh[..., 0:1] / 2,
        //    ys - wh[..., 1:2] / 2,
        //    xs + wh[..., 0:1] / 2,
        //    ys + wh[..., 1:2] / 2), axis = 1)
        //detections = np.concatenate((bboxes, scores, clses), axis = 1)
        //mask = detections[..., 4] >= self._threshold
        //filtered_detections = detections[mask]
        //scale = max(meta['original_shape'])
        //center = np.array(meta['original_shape'][:2]) / 2.0
        //dets = self._transform(filtered_detections, np.flip(center, 0), scale, height, width)
        //dets = [Detection(x[0], x[1], x[2], x[3], score = x[4], id = x[5]) for x in dets]
        //return dets
    auto heat = infResult.outputsData[outputsNames[0]];
    auto reg = infResult.outputsData[outputsNames[1]];
    auto wh = infResult.outputsData[outputsNames[2]];

}
